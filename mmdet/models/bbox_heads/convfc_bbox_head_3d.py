import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy

@HEADS.register_module
class SharedFCBBoxHead3D(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, loss_bbox_3d=None, with_reg_3d=True, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead3D, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.with_reg_3d = with_reg_3d
        if self.with_reg_3d:
            out_dim_reg_3d = (8 if self.reg_class_agnostic else 8 *
                                                         self.num_classes)
            self.fc_reg_3d = nn.Linear(self.reg_last_dim, out_dim_reg_3d)
            self.loss_bbox_3d = build_loss(loss_bbox_3d)


    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        if self.with_reg_3d:
            x_reg_3d = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_pred_3d = self.fc_reg_3d(x_reg_3d) if self.with_reg_3d else None
        # print(bbox_pred_3d)
        return cls_score, bbox_pred, bbox_pred_3d

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_bboxes_3d = [res.pos_gt_bboxes_3d for res in sampling_results]
        # print(pos_gt_bboxes[0].size(), pos_gt_bboxes_3d[0].size())
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        # logger.info('pos_gt_bboxes{}'.format(pos_gt_bboxes))
        # logger.info('pos_gt_bboxes_3d{}'.format(pos_gt_bboxes_3d[:2]))  # [[ 5.5125e-01,  9.7504e-01, -5.9984e-01,  3.7557e+02,  3.5705e+02, -1.3744e+00,  8.6807e-01,  1.0626e+00],
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            pos_gt_bboxes_3d=pos_gt_bboxes_3d)
        # logger.info('cls_reg_targets{}'.format(cls_reg_targets[3][:2]))
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_pred_3d,
             labels,
             label_weights,
             bbox_targets,
             bbox_targets_3d,
             bbox_weights,
             bbox_weights_3d,
             reduction_override=None):
        losses = dict()
        # print(bbox_pred.size(), bbox_pred_3d.size(), bbox_targets.size(), bbox_targets_3d.size())  # torch.Size([1024, 16]) torch.Size([1024, 32]) torch.Size([1024, 4]) torch.Size([1024, 8])
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0),
                                                   4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                                   4)[pos_inds,
                                                      labels[pos_inds]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

        if bbox_pred_3d is not None:
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred_3d = bbox_pred_3d.view(bbox_pred_3d.size(0),
                                                         8)[pos_inds]
                else:
                    pos_bbox_pred_3d = bbox_pred_3d.view(bbox_pred_3d.size(0), -1,
                                                         8)[pos_inds,
                                                            labels[pos_inds]]
                # from mmdet.apis import get_root_logger
                # logger = get_root_logger()
                # logger.info('pos_bbox_pred_3d{}'.format(pos_bbox_pred_3d))
                # logger.info('bbox_targets_3d{}'.format(bbox_targets_3d[pos_inds]))
                # logger.info('bbox_weights_3d{}'.format(bbox_weights_3d[pos_inds]))

                losses['loss_bbox_3d'] = self.loss_bbox_3d(
                    pos_bbox_pred_3d,
                    bbox_targets_3d[pos_inds],
                    bbox_weights_3d[pos_inds],
                    avg_factor=bbox_targets_3d.size(0),
                    reduction_override=reduction_override)
        return losses


    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_pred_3d'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       bbox_pred_3d,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes, bboxes_3d = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape, deltas_3d=bbox_pred_3d)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
                # from mmdet.apis import get_root_logger
                # logger = get_root_logger()
                # logger.info('2d_{}'.format(bboxes[:10, :]))
                # logger.info('old_{}'.format(bboxes_3d[:10, 2:6]))
                bboxes_3d[:, 3::8] /= scale_factor
                bboxes_3d[:, 4::8] /= scale_factor
                # logger.info('new_{}'.format(bboxes_3d[:10, 2:6]))
            else:
                scale_factor = torch.from_numpy(scale_factor).to(bboxes.device)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, bboxes_3d, scores
        else:
            det_bboxes, det_bboxes_3d, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, multi_bboxes_3d=bboxes_3d)

            return det_bboxes, det_bboxes_3d, det_labels