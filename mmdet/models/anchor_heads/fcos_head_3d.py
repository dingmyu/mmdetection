import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, distance2center, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import torch.nn.functional as F

import cv2
import numpy as np
INF = 1e8


def smooth_l1_loss(pred, target, beta=1.0/9.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@HEADS.register_module
class FCOSHead3D(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-INF, INF),),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_3d=dict(type='SmoothL1Loss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 std_3d = None):
        super(FCOSHead3D, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_3d = build_loss(loss_bbox_3d)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.std_3d = std_3d

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_convs_3d = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs_3d.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_reg_3d = nn.Conv2d(self.feat_channels, 8, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs_3d:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_reg_3d, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        reg_feat_3d = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        for reg_layer_3d in self.reg_convs_3d:
            reg_feat_3d = reg_layer_3d(reg_feat_3d)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float()#.exp()
        #bbox_pred = self.fcos_reg(reg_feat).float()
        bbox_pred_3d = self.fcos_reg_3d(reg_feat_3d).float()
        # print(scale.scale)
        return cls_score, bbox_pred, bbox_pred_3d, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_3d', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_3d,
             centernesses,
             gt_bboxes,
             gt_bboxes_3d,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_3d) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, bbox_targets_3d, bbox_center_3d, bbox_center_2d = self.fcos_target(all_level_points, gt_bboxes, gt_bboxes_3d,
                                                gt_labels)
        # print(labels[0].size(), bbox_targets[0].size())  # torch.Size([6784]) torch.Size([6784, 4]) batch*nunmber_point

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        if cfg.stat_2d:
            # from mmdet.apis import get_root_logger
            # logger = get_root_logger()
            stat = np.load('kitti_tools/stat/stat.npy').astype(np.float32)
            stat = cv2.resize(stat, (106, 32)).astype(np.float32)
            # stat = cv2.resize(stat, (4, 4)).astype(np.float32)
            # stat = cv2.resize(stat, (106, 32), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # std = np.std(stat, axis=(0, 1))
            # stat /= std
            z_stat = torch.from_numpy(stat[:,:,0]).float().cuda().unsqueeze(0)
            w_stat = torch.from_numpy(stat[:,:,1]).float().cuda().unsqueeze(0)
            h_stat = torch.from_numpy(stat[:,:,2]).float().cuda().unsqueeze(0)
            stat_all = torch.cat([w_stat, h_stat, w_stat, h_stat], dim = 0)
            # logger.info('old_{}'.format(bbox_preds_3d[0][0, :, 20:25, 20]))
            for bbox_pred_3d in bbox_preds_3d:
                bbox_pred_3d[:, 5:6, :, :] = bbox_pred_3d[:, 5:6, :, :] + z_stat
            bbox_preds = [bbox_pred * stat_all for bbox_pred in bbox_preds]  # torch.exp(bbox_pred)
            # logger.info('new_{}'.format(bbox_preds_3d[0][0, :, 20:25, 20]))
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_3d = [
            bbox_pred_3d.permute(0, 2, 3, 1).reshape(-1, 8)
            for bbox_pred_3d in bbox_preds_3d
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_3d = torch.cat(flatten_bbox_preds_3d)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_bbox_center_3d = torch.cat(bbox_center_3d)
        flatten_bbox_center_2d = torch.cat(bbox_center_2d)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        # logger.info(num_pos)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_3d = flatten_bbox_preds_3d[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]


        # check NaN and Inf
        # assert torch.isfinite(flatten_cls_scores).all().item(), \
        #     'classification scores become infinite or NaN!'
        assert torch.isfinite(pos_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'
        assert torch.isfinite(pos_bbox_preds_3d).all().item(), \
            'bbox_3d predications become infinite or NaN!'
        assert torch.isfinite(pos_centerness).all().item(), \
            'bbox centerness become infinite or NaN!'

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_bbox_center_3d = flatten_bbox_center_3d[pos_inds]
            pos_bbox_center_2d = flatten_bbox_center_2d[pos_inds]
            # print(pos_bbox_targets.size())  # torch.Size([669, 4])
            pos_centerness_targets = self.centerness_target(pos_bbox_center_2d, pos_bbox_center_3d)
            # print(pos_centerness_targets.size(), pos_centerness_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds, self.std_3d)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets, self.std_3d)
            # print(pos_bbox_targets, pos_bbox_targets_3d)
            # centerness weighted iou loss
            # print('target', pos_decoded_target_preds.size(), pos_decoded_target_preds)
            # print('pred', pos_decoded_bbox_preds.size(),pos_decoded_bbox_preds)
            # print((pos_decoded_target_preds - pos_decoded_bbox_preds).mean(0))
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            # print(pos_bbox_preds_3d.sum(1).size(), pos_centerness_targets.size())  # torch.Size([593]) torch.Size([593])
            # print('target', pos_bbox_targets_3d.size(), pos_bbox_targets_3d)
            # print('pred', pos_bbox_preds_3d.size(),pos_bbox_preds_3d)
            # from mmdet.apis import get_root_logger
            # logger = get_root_logger()
            # logger.info((pos_bbox_targets_3d - pos_bbox_preds_3d).mean(0))
            # loss_bbox_3d = self.loss_bbox_3d(
            #     pos_bbox_preds_3d,
            #     pos_bbox_targets_3d,
            #     weight=pos_centerness_targets
            # )
            loss_bbox_3d = (smooth_l1_loss(pos_bbox_preds_3d, pos_bbox_targets_3d).mean(1) * pos_centerness_targets).sum()/pos_centerness_targets.sum()
            # loss_bbox_3d = (F.smooth_l1_loss(pos_bbox_preds_3d, pos_bbox_targets_3d, reduction='none').sum(1) * pos_centerness_targets).mean()
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_3d = pos_bbox_preds_3d.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_3d=loss_bbox_3d,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_3d', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_3d,
                   centernesses,
                   img_metas,
                   # calib,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        # print(calib.size()) 1,4,4
        # calib = [calib]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list_3d= [
                bbox_preds_3d[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            # calib_list = [
            #     calib[i][img_id].detach() for i in range(num_levels)
            # ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list, bbox_pred_list_3d,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          bbox_preds_3d,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_bboxes_3d = []
        mlvl_scores = []
        mlvl_centerness = []
        if cfg.stat_2d:
            stat = np.load('kitti_tools/stat/stat.npy').astype(np.float32)
            stat = cv2.resize(stat, (106, 32)).astype(np.float32)
            # stat = cv2.resize(stat, (4, 4)).astype(np.float32)
            # stat = cv2.resize(stat, (106, 32), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # std = np.std(stat, axis=(0, 1))
            # stat /= std
            z_stat = torch.from_numpy(stat[:,:,0]).float().cuda().unsqueeze(0)
            w_stat = torch.from_numpy(stat[:,:,1]).float().cuda().unsqueeze(0)
            h_stat = torch.from_numpy(stat[:,:,2]).float().cuda().unsqueeze(0)
            stat_all = torch.cat([w_stat, h_stat, w_stat, h_stat], dim = 0)
            for bbox_pred_3d in bbox_preds_3d:
                bbox_pred_3d[5:6, :, :] = bbox_pred_3d[5:6, :, :] + z_stat
            bbox_preds = [bbox_pred * stat_all for bbox_pred in bbox_preds]  # torch.exp(bbox_pred)
        for cls_score, bbox_pred, bbox_pred_3d,centerness, points in zip(
                cls_scores, bbox_preds, bbox_preds_3d, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1)  # .sigmoid()  # TODO: MINGYU if sigmoid?

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred_3d = bbox_pred_3d.permute(1, 2, 0).reshape(-1, 8)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                bbox_pred_3d = bbox_pred_3d[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, self.std_3d, max_shape=img_shape)
            bboxes_3d = distance2center(points, bbox_pred_3d, self.std_3d)
            mlvl_bboxes.append(bboxes)
            mlvl_bboxes_3d.append(bboxes_3d)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_3d = torch.cat(mlvl_bboxes_3d)
        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        if rescale:
            # logger.info(mlvl_bboxes_3d[:, 3:5])
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)  # TODO: add 3D
            mlvl_bboxes_3d[:, 3:5] /= mlvl_bboxes_3d[:, 3:5].new_tensor(scale_factor)
            # logger.info('new_{}'.format(mlvl_bboxes_3d[:, 3:5]))
            # print(scale_factor, mlvl_bboxes_3d.size())  # 1.3653333333333333 torch.Size([1000, 8])
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_bboxes_3d, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            multi_bboxes_3d=mlvl_bboxes_3d)
        return det_bboxes, det_bboxes_3d, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_bboxes_list_3d, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # print(expanded_regress_ranges)  # [-100000000.,  100000000.]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # print(concat_regress_ranges.size(), concat_points.size())  # torch.Size([3392, 2]) torch.Size([3392, 2])
        # get labels and bbox_targets of each image
        # print(gt_bboxes_list_3d, gt_bboxes_list, gt_labels_list)  # n,8   n,4   n  original
        labels_list, bbox_targets_list, bbox_targets_list_3d, bbox_center_list_3d, bbox_center_list_2d = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_bboxes_list_3d,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)  # torch.split(tensor, split_size, dim=0)
            for bbox_targets in bbox_targets_list
        ]
        bbox_targets_list_3d = [
            bbox_targets_3d.split(num_points, 0)  # torch.split(tensor, split_size, dim=0)
            for bbox_targets_3d in bbox_targets_list_3d
        ]
        bbox_center_list_3d = [
            bbox_center_3d.split(num_points, 0)  # torch.split(tensor, split_size, dim=0)
            for bbox_center_3d in bbox_center_list_3d
        ]
        bbox_center_list_2d = [
            bbox_center_2d.split(num_points, 0)  # torch.split(tensor, split_size, dim=0)
            for bbox_center_2d in bbox_center_list_2d
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_bbox_center_3d = []
        concat_lvl_bbox_center_2d = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_bbox_targets_3d.append(
                torch.cat(
                    [bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_list_3d]))
            concat_lvl_bbox_center_3d.append(
                torch.cat(
                    [bbox_center_3d[i] for bbox_center_3d in bbox_center_list_3d]))
            concat_lvl_bbox_center_2d.append(
                torch.cat(
                    [bbox_center_2d[i] for bbox_center_2d in bbox_center_list_2d]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bbox_targets_3d, concat_lvl_bbox_center_3d, concat_lvl_bbox_center_2d

    def fcos_target_single(self, gt_bboxes, gt_bboxes_3d, gt_labels, points, regress_ranges):
        # print(gt_bboxes.size(), gt_labels.size(), points.size(), regress_ranges.size())
        # torch.Size([7, 4])
        # torch.Size([7])
        # torch.Size([3392, 2]) 32*106
        # torch.Size([3392, 2])
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes_3d.new_zeros((num_points, 8))

        # areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
        #     gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        # areas = areas[None].repeat(num_points, 1)
        # print('areas', areas.size())  # areas torch.Size([3392, 6])
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        # print(regress_ranges.size())  # 3392, 7, 2
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        areas = gt_bboxes_3d[:, 5][None].repeat(num_points, 1)  # ~~~~~ depth
        # print(gt_bboxes_3d[..., -3].size(), gt_bboxes_3d[..., -3])  torch.Size([3]) tensor([-1.1724, -1.5243, -0.3568], device='cuda:0')
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts, 8)
        # print(gt_bboxes_3d.size(), gt_bboxes_3d[..., 3:])  # torch.Size([3392, 2, 8]) tensor([[[ 5.1355e+02,  3.0273e+02, -1.3587e+00, -3.1283e+00, -2.9301e+00],
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = (xs - gt_bboxes[..., 0])/ (self.std_3d[0]*1.365) -xs.new_tensor(1.0)  # original size, 512*1696
        right = (gt_bboxes[..., 2] - xs)/ (self.std_3d[0]*1.365) -xs.new_tensor(1.0)
        top = (ys - gt_bboxes[..., 1])/ (self.std_3d[1]*1.365) -xs.new_tensor(1.0)
        bottom = (gt_bboxes[..., 3] - ys)/ (self.std_3d[1]*1.365) -xs.new_tensor(1.0)
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # print(bbox_targets.size(), bbox_targets)  # torch.Size([3392, 6, 4]) tensor([[[  2.3096,  -3.1293,   6.2606,   7.9404],


        left = (xs - gt_bboxes[..., 0])  # original size, 512*1696
        right = (gt_bboxes[..., 2] - xs)
        top = (ys - gt_bboxes[..., 1])
        bottom = (gt_bboxes[..., 3] - ys)
        bbox_center_2d = torch.stack((left, top, right, bottom), -1)

        left = gt_bboxes_3d[..., 3] - gt_bboxes[..., 0]  # original size, 512*1696
        right = gt_bboxes[..., 2] - gt_bboxes_3d[..., 3]
        top = gt_bboxes_3d[..., 4] - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - gt_bboxes_3d[..., 4]
        bbox_center_3d = torch.stack((left, top, right, bottom), -1)
        # print(bbox_center_3d.size(), bbox_center_3d)  # torch.Size([3392, 5, 4]) tensor([[[56.3121, 39.2054, 48.3832, 47.5418],

        left = torch.stack((bbox_center_2d[..., 0], bbox_center_3d[..., 0]), -1)
        right = torch.stack((bbox_center_2d[..., 2], bbox_center_3d[..., 2]), -1)
        top = torch.stack((bbox_center_2d[..., 1], bbox_center_3d[..., 1]), -1)
        bottom = torch.stack((bbox_center_2d[..., 3], bbox_center_3d[..., 3]), -1)

        left_right = (left.min(dim=-1)[0] / left.max(dim=-1)[0]) * (right.min(dim=-1)[0] / right.max(dim=-1)[0])
        top_bottom = (top.min(dim=-1)[0] / top.max(dim=-1)[0]) * (bottom.min(dim=-1)[0] / bottom.max(dim=-1)[0])

        # areas = left_right * top_bottom
        inside_gt_center_mask =  (left_right * top_bottom > 0.3)

        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        # logger.info('old_{}'.format(gt_bboxes_3d[..., 3]))
        # print(gt_bboxes_3d[..., 3].size(), xs.size())  # torch.Size([3392, 5]) torch.Size([3392, 5])
        # logger.info('xs{}'.format(xs))
        # logger.info('3d{}'.format(gt_bboxes_3d[..., 3] - xs))
        # gt_bboxes_3d[..., 3] = gt_bboxes_3d[..., 3] - xs  # X_p, Y_p
        # gt_bboxes_3d[..., 4] = gt_bboxes_3d[..., 4] - ys
        # gt_bboxes_3d[..., 3] = gt_bboxes_3d[..., 3]  # TODO: use variable
        # gt_bboxes_3d[..., 4] = gt_bboxes_3d[..., 4]
        # print(gt_bboxes_3d[..., 3:6])
        center_x = ((gt_bboxes_3d[..., 3] - xs)/ (self.std_3d[0]*1.365) - 0.016801083)/0.6852198
        center_y = ((gt_bboxes_3d[..., 4] - ys)/ (self.std_3d[1]*1.365) + 0.03962014)/0.38124686

        bbox_targets_3d = torch.stack((gt_bboxes_3d[..., 0],gt_bboxes_3d[..., 1],gt_bboxes_3d[..., 2],
                                    center_x,center_y,gt_bboxes_3d[..., 5],
                                    gt_bboxes_3d[..., 6],gt_bboxes_3d[..., 7]), -1)
        # print(gt_bboxes_3d[..., 3:6])
        # logger.info('new_{}'.format(gt_bboxes_3d[..., 3]))

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_center_2d.min(-1)[0] > 0#) & (bbox_center_3d.min(-1)[0] > 0)
        # print(inside_gt_bbox_mask.size(), inside_gt_bbox_mask.float().mean())  # torch.Size([3392, 6]) tensor(0.0074, device='cuda:0')
        # inside_gt_center_mask = bbox_targets.min(-1)[0] > bbox_targets.max(-1)[0]/3  # ~~~~~~ center
        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        # logger.info(bbox_targets.min(-1)[0], bbox_targets.max(-1)[0])

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_center_2d.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # print(inside_regress_range.size(), inside_regress_range.float().mean(), inside_regress_range)  1

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_gt_center_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        # print(gt_bboxes_3d[..., -3].size(), gt_bboxes_3d[..., -3].mean(0))  torch.Size([3392, 3]) tensor([-1.1724, -1.5243, -0.3568], device='cuda:0')
        min_area, min_area_inds = areas.min(dim=1)  # TODO: max or min

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_area_inds]
        bbox_center_3d = bbox_center_3d[range(num_points), min_area_inds]
        bbox_center_2d = bbox_center_2d[range(num_points), min_area_inds]
        # print(labels.size(), bbox_targets.size())  # torch.Size([3392]) torch.Size([3392, 4])
        return labels, bbox_targets, bbox_targets_3d, bbox_center_3d, bbox_center_2d

    def centerness_target(self, pos_bbox_targets, pos_bbox_center_3d):  # calculate for each feature pixel according to their GT
        # only calculate pos centerness targets, otherwise there may be nan
        # print(pos_bbox_targets.size(), pos_bbox_center_3d.size())  # torch.Size([458, 4]) torch.Size([458, 4])
        left = torch.stack((pos_bbox_targets[:, 0], pos_bbox_center_3d[:, 0]), -1)
        right = torch.stack((pos_bbox_targets[:, 2], pos_bbox_center_3d[:, 2]), -1)
        top = torch.stack((pos_bbox_targets[:, 1], pos_bbox_center_3d[:, 1]), -1)
        bottom = torch.stack((pos_bbox_targets[:, 3], pos_bbox_center_3d[:, 3]), -1)

        left_right = (left.min(dim=-1)[0] / left.max(dim=-1)[0]) * (right.min(dim=-1)[0] / right.max(dim=-1)[0])
        top_bottom = (top.min(dim=-1)[0] / top.max(dim=-1)[0]) * (bottom.min(dim=-1)[0] / bottom.max(dim=-1)[0])
        # from mmdet.apis import get_root_logger
        # logger = get_root_logger()
        # logger.info('my_{}'.format(left_right * top_bottom))
        # left_right = pos_bbox_targets[:, [0, 2]]
        # top_bottom = pos_bbox_targets[:, [1, 3]]
        # centerness_targets = (
        #     left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        #         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        # logger.info(centerness_targets)
        # return torch.sqrt(centerness_targets)
        # return torch.sqrt(left_right * top_bottom)
        return left_right * top_bottom
