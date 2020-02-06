from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
import torch

@DETECTORS.register_module
class FCOS3D(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_bboxes_3d,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_bboxes_3d, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, calib, rescale=False):
        mean_3d = [1.566141, 1.4557937, 3.393441, 0, 0, 26.497492, 1.5803262, 0.528173]
        std_3d = [0.15824416, 0.39049828, 1.1481832, 1, 1, 16.059835, 0.678825, 0.49920323]
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        for i_3d in range(8):
            bbox_list[0][1][:, i_3d] = bbox_list[0][1][:, i_3d] * std_3d[i_3d] + mean_3d[i_3d]

        calib_inv = calib[0].inverse()
        x = bbox_list[0][1][:, 3]
        y = bbox_list[0][1][:, 4]
        z = bbox_list[0][1][:, 5]
        position_3d = torch.stack((x*z, y*z, z, torch.ones_like(z)), -1)
        position_3d = calib_inv.mm(position_3d.t())[:3,:].t()
        bbox_list[0][1][:, 3:6] = position_3d
        # print(bbox_list[0][0].size()) #torch.Size([59, 5])
        # print(bbox_list[0][1].size()) #torch.Size([59, 8])
        # print(bbox_list[0][2].size()) #torch.Size([59])
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes, bboxes_3d=det_bboxes_3d)
            for det_bboxes, det_bboxes_3d, det_labels in bbox_list
        ]
        return bbox_results[0]

    def forward_test(self, imgs, img_metas, calib, rescale=False):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], calib[0], rescale)
        else:
            return self.aug_test(imgs, img_metas)