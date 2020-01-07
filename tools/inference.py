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

config_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/BASELINE_lr_0.002_nms_0.4_epoch_24_2020_01_06_17_12_26/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/mnt/lustre/dingmingyu/2020/mmdetection/output/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/BASELINE_lr_0.002_nms_0.4_epoch_24_2020_01_06_17_12_26/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
for i in range(7000):
    img = 'data/kitti/testing/image_2/%06d.png' % i
    result = inference_detector(model, img)
    img = show_result(
        img, result, model.CLASSES, score_thr=0.3, show=False)
    cv2.imwrite('output/visualization/%06d.png' % i, img)