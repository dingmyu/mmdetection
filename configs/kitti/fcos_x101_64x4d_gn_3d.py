# model settings
import datetime


total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5)
workflow = [('train', 1)]
dist_params = dict(backend='nccl', port=9899)
log_level = 'INFO'
name = 'FINAL'

card = 8
pretrain = True
stat_2d = False
channels = 3

if pretrain:
    load_from = '/mnt/lustre/dingmingyu/2020/mmdetection/work_dirs/20200128_011929_2D_baseline_lr_0.001_nms_0.4_epoch_12/epoch_12.pth'
    # load_from = './output/fcos_x101_64x4d_gn_coco/20200208_010041_2D_baseline_lr_0.001_nms_0.4_epoch_10/latest.pth'
else:
    load_from = None
resume_from = None


copy_dict = dict(
    FCOS='mmdet/models/detectors/fcos.py',
    FCOSHead2D='mmdet/models/anchor_heads/fcos_head_2d.py',
    FCOS3D='mmdet/models/detectors/fcos_3d.py',
    FCOSHead3D='mmdet/models/anchor_heads/fcos_head_3d.py',
)

model = dict(
    type='FCOS3D',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='Dilated_ResNeXt',
        in_channels=channels,
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='FCOSHead3D',
        num_classes=4,
        in_channels=2048,
        stacked_convs=2,
        feat_channels=512,
        strides=[16,],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_bbox_3d=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=2.0),
        loss_centerness=dict(
            #type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
            type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        std_3d = [123.62374/2, 113.182/2]
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    stat_2d=stat_2d)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.5,
    mean_3d= [-0.00378265,  -0.00182043,  -0.00622482, 0, 0, 26.235588,  -0.04752721,  -0.02627299],
    std_3d = [0.08565227,   0.06028508,   0.11255758, 1, 1, 12.873396,  1.6905682,  1.7750752],
    nms=dict(type='nms', iou_thr=0.4),
    max_per_img=100,
    stat_2d=stat_2d)
# dataset settings
dataset_type = 'Kitti3dDataset'
data_root = 'kitti_tools/split1/'
if channels == 6:
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # RGB + 2d_wh
else:
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', channels=channels),
    dict(type='LoadAnnotations', with_bbox=True, with_bbox_3d=True),
    dict(
        type='Resize',
        img_scale=(1696, 512),
        #multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_3d', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', channels=channels),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1696, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'calib']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_3d.pkl',
        img_prefix=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_3d.pkl',
        img_prefix=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_3d.pkl',
        img_prefix=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='onecycle',)
    # warmup='constant',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    # step=[16, 22])
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings

alias = '_'.join([name, 'pretrain', str(pretrain), 'stat', str(stat_2d), 'lr', lr_config['policy'], str(optimizer['lr']), 'nms', str(test_cfg['nms']['iou_thr']), 'epoch', str(total_epochs), 'batch', str(data['imgs_per_gpu'] * card)])
