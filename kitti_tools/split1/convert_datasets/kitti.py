# Author: Mingyu Ding
# Time: 2/1/2020 7:02 PM
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

from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pprint
import sys
import os
import cv2
import math
import shutil
import re
from easydict import EasyDict as edict
import json
import pickle
import mmcv

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)


def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.
    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2


def read_kitti_label(file, dataset):
    """
    Reads the kitti label file from disc.
    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''

    pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                          + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                         .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))

    bboxes = []
    bboxes_ignore = []
    labels = []
    labels_ignore = []

    TYPE2LABEL = dict(
        Background=0,
        Car=1,
        Cyclist=2,
        Pedestrian=3,
        # Van=4,
        # Person_sitting=5
        # Truck = 6
        # Tram = 7
        # Misc = 8
    )

    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            label_type = parsed.group(1)  # type
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = float(parsed.group(5))  # left
            y = float(parsed.group(6))  # top
            x2 = float(parsed.group(7))  # right
            y2 = float(parsed.group(8))  # bottom

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(9))
            w3d = float(parsed.group(10))
            l3d = float(parsed.group(11))

            cx3d = float(parsed.group(12))  # center of car in 3d
            cy3d = float(parsed.group(13))  # bottom of car in 3d
            cz3d = float(parsed.group(14))  # center of car in 3d
            rotY = float(parsed.group(15))

            obj.bbox_3d = [w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            obj.bbox_2d = [x, y, x2, y2]
            obj.label = label_type

            gts.append(obj)

            if label_type in TYPE2LABEL:
                bboxes.append(obj.bbox_2d)
                labels.append(TYPE2LABEL[label_type])
            if label_type == 'DontCare':
                bboxes_ignore.append(obj.bbox_2d)
                labels_ignore.append(0)

            if dataset == 'train':
                if bboxes_ignore:
                    ann = dict(
                        bboxes=np.array(bboxes, dtype=np.float32),
                        labels=np.array(labels, dtype=np.int64),
                        bboxes_ignore=np.array(bboxes_ignore, dtype=np.float32),
                        #                     labels_ignore=np.array(labels_ignore, dtype=np.int64)
                    )
                else:
                    ann = dict(
                        bboxes=np.array(bboxes, dtype=np.float32),
                        labels=np.array(labels, dtype=np.int64),
                        bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
                        #                     labels_ignore=np.zeros((0, ))
                    )
            else:
                ann = dict(
                    bboxes=np.array(bboxes, dtype=np.float32),
                    labels=np.array(labels, dtype=np.int64),
                )

    return gts, ann



kitti_raw = dict()
kitti_raw['base'] = os.path.join(os.getcwd(), 'data', 'kitti')
kitti_raw['calib'] = os.path.join(kitti_raw['base'], 'training', 'calib')
kitti_raw['img'] = os.path.join(kitti_raw['base'], 'training', 'image_2')
kitti_raw['label'] = os.path.join(kitti_raw['base'], 'training', 'label_2')
kitti_raw['pre'] = os.path.join(kitti_raw['base'], 'training', 'prev_2')

train_file = 'kitti_tools/split1/train.txt'
val_file = 'kitti_tools/split1/val.txt'
train_list = []
val_list = []

for item in ['train', 'val']:
    if item == 'train':
        text_file = open(train_file, 'r')
    else:
        text_file = open(val_file, 'r')

    for index, line in enumerate(text_file):
        if index % 20 == 0:
            print(index)
        parsed = re.search('(\d+)', line)
        if parsed is not None:
            id = str(parsed[0])

            file_info = dict()
            file_info['filename'] = os.path.join(kitti_raw['img'], id + '.png')
            file_info['calibname'] = os.path.join(kitti_raw['calib'], id + '.txt')
            file_info['labelname'] = os.path.join(kitti_raw['label'], id + '.txt')

            file_shape = cv2.imread(file_info['filename']).shape[:2][::-1]
            file_info['width'] = file_shape[0]
            file_info['height'] = file_shape[1]
            file_info['calib'] = read_kitti_cal(file_info['calibname'])
            file_info['label'], file_info['ann'] = read_kitti_label(file_info['labelname'], item)
            if item == 'train':
                train_list.append(file_info)
            else:
                val_list.append(file_info)
# output = open(os.path.join(os.getcwd(), 'train.pkl'), 'wb')
# pickle.dump(train_list, output)
mmcv.dump(train_list, os.path.join(os.getcwd(), os.path.join(os.getcwd(), 'kitti_tools', 'split1', 'train.pkl')))
mmcv.dump(val_list, os.path.join(os.getcwd(), os.path.join(os.getcwd(), 'kitti_tools', 'split1', 'val.pkl')))