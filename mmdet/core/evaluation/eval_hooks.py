import re
import os
import os.path as osp

import mmcv
import numpy as np
import subprocess
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet import datasets
from .coco_utils import fast_eval_recall, results2json
from .mean_ap import eval_map


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if 'bboxes_ignore' in ann:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        if not gt_ignore:
            gt_ignore = None
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


def parse_kitti_result(respath, mode='new'):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall(r'([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    if mode == 'old':
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])
    else:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])

    return easy, mod, hard


class KITTIDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        from mmdet.apis import get_root_logger
        mean_3d = [1.4558165, 1.5661651, 3.3934565, 26.494734]
        std_3d = [0.39049387, 0.15824589, 1.1481881, 16.059843]
        logger = get_root_logger()
        ds_name = self.dataset.CLASSES
        mmcv.mkdir_or_exist(
            osp.join(runner.work_dir, 'epoch_{}'.format(str(runner.epoch + 1)), 'data'))
        for i in range(len(self.dataset)):
            # file_name = self.dataset.get_file_name(i).split('/')[-1].replace('png', 'txt')
            f = open(osp.join(runner.work_dir, 'epoch_{}'.format(
                str(runner.epoch + 1)), 'data', '%06d.txt' % i), 'w')
            for category, result in enumerate(results[i]):
                if result.any():
                    for item in result:
                        if len(item) == 5:
                            print(ds_name[category], -1, -1, 0, item[0], item[1], item[2], item[3], 0, 0, 0, 0, 0, 0, 0, item[4], file=f)
                        if len(item) == 5 + 8:
                            for i_3d in range(5, 8):
                                item[i_3d] = item[i_3d] * std_3d[i_3d - 5] + mean_3d[i_3d - 5]
                            item[10] = item[10] * std_3d[3] + mean_3d[3]
                            if item[12] > 0.5:
                                item[11] = -item[11]
                            print(ds_name[category], -1, -1, 0, item[0], item[1], item[2], item[3], item[5], item[6], item[7], item[8], item[9], item[10], item[11], item[4], file=f)
            f.close()

        script = os.path.join(
            os.getcwd(),
            'kitti_tools',
            'split1',
            'devkit',
            'cpp',
            'evaluate_object')
        with open(os.devnull, 'w') as devnull:
            out = subprocess.check_output([script, osp.join(
                runner.work_dir, 'epoch_{}'.format(str(runner.epoch + 1)))], stderr=devnull)

        results_path = osp.join(runner.work_dir, 'epoch_{}'.format(
            str(runner.epoch + 1)), 'data')
        test_iter = 0

        for lbl in ['Car', 'Cyclist', 'Pedestrian']:

            lbl = lbl.lower()

            respath_2d = os.path.join(results_path.replace(
                '/data', ''), 'stats_{}_detection.txt'.format(lbl))
            respath_gr = os.path.join(
                results_path.replace(
                    '/data',
                    ''),
                'stats_{}_detection_ground.txt'.format(lbl))
            respath_3d = os.path.join(
                results_path.replace(
                    '/data',
                    ''),
                'stats_{}_detection_3d.txt'.format(lbl))

            if os.path.exists(respath_2d):
                easy, mod, hard = parse_kitti_result(respath_2d, mode='old')

                print_str = 'R11_test_epoch {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)
                logger.info(print_str)

                easy, mod, hard = parse_kitti_result(respath_2d)

                print_str = 'R40_test_epoch {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)
                logger.info(print_str)

            if os.path.exists(respath_gr):
                easy, mod, hard = parse_kitti_result(respath_gr, mode='old')

                print_str = 'R11_test_epoch {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)

                logger.info(print_str)

                easy, mod, hard = parse_kitti_result(respath_gr)

                print_str = 'R40_test_epoch {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)

                logger.info(print_str)

            if os.path.exists(respath_3d):
                easy, mod, hard = parse_kitti_result(respath_3d, mode='old')

                print_str = 'R11_test_epoch {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)

                logger.info(print_str)

                easy, mod, hard = parse_kitti_result(respath_3d)

                print_str = 'R40_test_epoch {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
                    runner.epoch + 1, lbl, easy, mod, hard)

                logger.info(print_str)


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'
                     ] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            try:
                cocoDt = cocoGt.loadRes(result_files[res_type])
            except IndexError:
                print('No prediction found.')
                break
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])
