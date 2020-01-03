# Author: Mingyu Ding
# Time: 2/1/2020 9:33 PM
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

import re
import numpy as np
import os


def parse_kitti_result(respath, mode='new'):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

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


results_path = '/mnt/lustre/dingmingyu/2020/mmdetection/work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/name_2019_10_20/epoch_1/data'  # TODO
test_iter = 0


for lbl in ['Car', 'Cyclist', 'Pedestrian']:

    lbl = lbl.lower()

    respath_2d = os.path.join(results_path.replace('/data', ''), 'stats_{}_detection.txt'.format(lbl))
    respath_gr = os.path.join(results_path.replace('/data', ''), 'stats_{}_detection_ground.txt'.format(lbl))
    respath_3d = os.path.join(results_path.replace('/data', ''), 'stats_{}_detection_3d.txt'.format(lbl))

    if os.path.exists(respath_2d):
        easy, mod, hard = parse_kitti_result(respath_2d, mode='old')

        print_str = 'OLD_test_iter {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)
        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_2d)

        print_str = 'NEW_test_iter {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)
        print(print_str)

    if os.path.exists(respath_gr):
        easy, mod, hard = parse_kitti_result(respath_gr, mode='old')

        print_str = 'OLD_test_iter {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)

        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_gr)

        print_str = 'NEW_test_iter {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)

        print(print_str)

    if os.path.exists(respath_3d):
        easy, mod, hard = parse_kitti_result(respath_3d, mode='old')

        print_str = 'OLD_test_iter {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)

        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_3d)

        print_str = 'NEW_test_iter {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, lbl,
                                                                                                   easy, mod, hard)

        print(print_str)
