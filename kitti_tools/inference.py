# Author: Mingyu Ding
# Time: 6/1/2020 8:42 PM
# Copyright 2019. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2

config_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/fcos_x101_64x4d_gn/20200124_043725_2D_baseline_lr_0.01_nms_0.4_epoch_80/fcos_x101_64x4d_gn.py'
checkpoint_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/fcos_x101_64x4d_gn/20200124_043725_2D_baseline_lr_0.01_nms_0.4_epoch_80/epoch_52.pth'
# config_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/faster_rcnn_x101_64x4d_fpn_1x/2020_01_10_10_59_40_2D_FPN_lr_0.02_nms_0.5_epoch_24/faster_rcnn_x101_64x4d_fpn_1x.py'
# checkpoint_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/faster_rcnn_x101_64x4d_fpn_1x/2020_01_10_10_59_40_2D_FPN_lr_0.02_nms_0.5_epoch_24/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
f = open('kitti_tools/split1/val.txt').readlines()
for i, line in enumerate(f):
    if i > 200:
        break
    if i % 10 == 0:
        print(i)
    line = int(line.strip())
    img = 'data/kitti/training/image_2/%06d.png' % line
    result = inference_detector(model, img)
    img = show_result(
        img, result, model.CLASSES, score_thr=0.3, show=False)
    cv2.imwrite('output/visualization/2d_onecycle/%06d.png' % line, img)