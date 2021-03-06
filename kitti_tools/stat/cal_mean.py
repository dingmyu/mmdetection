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

mean_3d = [ 1.5319942,    1.6342136,    3.8970344,  1.0385063, -2.2421432, 26.235588,  -0.04752721,  -0.02627299]
std_3d = [ 0.13790289,   0.09824956,   0.42854765, 42.354717, 21.57514, 12.873396,  1.6905682,  1.7750752]
Aw = 123.62374  # 69.16813
Ah = 113.182  # 51.076797]

trans_mean = [-0.00378265,  -0.00182043,  -0.00622482, 0.016801083, -0.03962014, 0.00015847,   0.00000034,  -0.00000001]
trans_std = [0.08565227,   0.06028508,   0.11255758, 0.6852198, 0.38124686, 12.873393,  1.6905676,   1.7750752]



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

def convertRot2Alpha(ry3d, z3d, x3d):

    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
    # alpha = ry3d - math.atan2(x3d, z3d)  # equivalent

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices
    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate object coordinate to camera coordinate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]  # normalize dim 3 -> 2, image coordinate

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d  # 2d corners in image coordinate and 3d corners in camera coordinate
    else:
        return verts3d

def read_kitti_label(file, calib, dataset):
    """
    Reads the kitti label file from disc.
    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []
    calib = calib.astype(np.float32)

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
    bboxes_3d = []
    bboxes_3d_ignore = []
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

            cy3d -= (h3d / 2)

            ign = 0
            verts3d, corners_3d = project_3d(calib, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

            if np.any(corners_3d[2, :] <= 0):
                ign = True
            else:  # 3d for 2d
                x = min(verts3d[:, 0])
                y = min(verts3d[:, 1])
                x2 = max(verts3d[:, 0])
                y2 = max(verts3d[:, 1])

                width = x2 - x + 1
                height = y2 - y + 1

            # rotY_1 = 0
            # if rotY < 0:
            #     rotY = rotY + math.pi
            #     rotY_1 = 1

            x_p, y_p, z_p, _ = calib.dot(np.array([cx3d, cy3d, cz3d, 1]))
            x_p /= z_p
            y_p /= z_p

            alpha = convertRot2Alpha(rotY, cz3d, cx3d)
            # obj.bbox_3d = [h3d, w3d, l3d, x_p, y_p, z_p, rotY, alpha, (x_p - x) / 2, -(x2 - x_p) / 2, (y_p - y) / 2,
            #                -(y2 - y_p) / 2]
            obj.bbox_3d = [np.log(h3d/mean_3d[0]), np.log(w3d/mean_3d[1]), np.log(l3d/mean_3d[2]), x_p, y_p, z_p - mean_3d[5], rotY - mean_3d[6], alpha - mean_3d[7], (x_p - x)/Aw, -(x2 - x_p)/Aw, (y_p - y)/Ah, -(y2- y_p)/Ah]

            obj.bbox_2d = [width, height]
            obj.label = label_type

            gts.append(obj)

            if label_type in ['Car'] and occ <= 1 and not ign and height >= 23 and height <= 280:
                bboxes.append(obj.bbox_2d)
                bboxes_3d.append(obj.bbox_3d)
                labels.append(TYPE2LABEL[label_type])

    return np.array(bboxes_3d, dtype=np.float32), np.array(bboxes, dtype=np.float32)



kitti_raw = dict()
kitti_raw['base'] = os.path.join(os.getcwd(), 'data', 'kitti')
kitti_raw['calib'] = os.path.join(kitti_raw['base'], 'training', 'calib')
kitti_raw['img'] = os.path.join(kitti_raw['base'], 'training', 'image_2')
kitti_raw['label'] = os.path.join(kitti_raw['base'], 'training', 'label_2')
kitti_raw['pre'] = os.path.join(kitti_raw['base'], 'training', 'prev_2')

train_list = []
train_list_2d = []

# stats = np.zeros((0, 8), np.float32)


for id in range(7481):
    if id % 20 == 19:
        print(id)
        # break
    id = '%06d' % id
    file_info = dict()
    file_info['filename'] = os.path.join(kitti_raw['img'], id + '.png')
    file_info['calibname'] = os.path.join(kitti_raw['calib'], id + '.txt')
    file_info['labelname'] = os.path.join(kitti_raw['label'], id + '.txt')

    file_shape = cv2.imread(file_info['filename']).shape[:2][::-1]
    file_info['width'] = file_shape[0]
    file_info['height'] = file_shape[1]

    file_info['calib'] = read_kitti_cal(file_info['calibname'])
    bbox_3d, bbox_2d = read_kitti_label(file_info['labelname'], file_info['calib'], '1')
    train_list.extend(bbox_3d)
    train_list_2d.extend(bbox_2d)
xx = np.array(train_list)
print(xx.mean(0), xx.std(0))
print(xx[:,-4:-2].mean(), xx[:,-4:-2].std())
print(xx[:, -2:].mean(), xx[:, -2:].std())

xx_2d = np.array(train_list_2d)
print(xx_2d.mean(0), xx_2d.std(0))

# stats = np.concatenate((stats, file_info['ann']['bboxes_3d']), axis=0)
    # print(stats.mean(0), stats.std(0))
# output = open(os.path.join(os.getcwd(), 'train.pkl'), 'wb')
# pickle.dump(train_list, output)
# print(stats.mean(0),stats.std(0))
