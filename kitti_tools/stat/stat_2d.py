# Author: Mingyu Ding
# Time: 28/1/2020 6:13 PM
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

import easydict as edict
def read_kitti_label(file):
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
    all_boxes = []
    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

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
            # all_boxes.append(((x+x2)/2, (y+y2)/2, width, height))
            all_boxes.append((x, x2, y, y2, width, height))
    return all_boxes

import re
import math
import os
import numpy as np
all_boxes = []
for index in range(7481):
    if index % 500 == 0:
        print(index)
#     if index >= 100:
#         break
    id = '%06d' % index
    labelname = os.path.join('/mnt/lustre/dingmingyu/2020/mmdetection/', 'data', 'kitti', 'training', 'label_2', id + '.txt')
    label = read_kitti_label(labelname)
    all_boxes.extend(label)


a = np.zeros((375,1242,3), np.float)
for item in all_boxes:
    x, x2, y, y2, w, h = item
    x, y = int(x), int(y)
    x2, y2 = math.ceil(x2), math.ceil(y2)
    w, h = float(w), float(h)
    a[y:y2, x:x2, 0] += 1
    a[y:y2, x:x2, 1] += w
    a[y:y2, x:x2, 2] += h

a[:,:,0][a[:,:,0]==0] = 1
a[:,:,1] /= a[:,:,0]
a[:,:,2] /= a[:,:,0]

a[:,:,0][a[:,:,1]==0] = 0
a[:,:,0][a[:,:,1]!=0] = 1

print('number of None:', (a[:,:,0]==0).sum()/(1242*375))

# import matplotlib.pyplot as plt
# plt.imshow(a.astype(np.int))

a[0,:,1][a[0,:,1]==0] = a[0,:,1][a[0,:,1]!=0].mean()
a[0,:,2][a[0,:,2]==0] = a[0,:,2][a[0,:,2]!=0].mean()
a[-1,:,1][a[-1,:,1]==0] = a[-1,:,1][a[-1,:,1]!=0].mean()
a[-1,:,2][a[-1,:,2]==0] = a[-1,:,2][a[-1,:,2]!=0].mean()
a[:,0,1][a[:,0,1]==0] = a[:,0,1][a[:,0,1]!=0].mean()
a[:,0,2][a[:,0,2]==0] = a[:,0,2][a[:,0,2]!=0].mean()
a[:,-1,1][a[:,-1,1]==0] = a[:,-1,1][a[:,-1,1]!=0].mean()
a[:,-1,2][a[:,-1,2]==0] = a[:,-1,2][a[:,-1,2]!=0].mean()
# a[:,:,1][a[:,:,1]==0] = a[0,:,1][a[0,:,1]!=0].mean()
# a[:,:,2][a[:,:,2]==0] = a[0,:,2][a[0,:,2]!=0].mean()

while (a[:,:,1]==0).any():
    a[:,:,0][a[:,:,1]==0] = 0
    a[:,:,0][a[:,:,1]!=0] = 1
    print('ok')
    b = np.zeros((375-2,1242-2,3), np.float)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            b += a[i+1:375+i-1, j+1:1242+j-1, :]
    b[:,:,0][b[:,:,0]==0] = 1
    b[:,:,1:] /= b[:,:,0:1]

    a[1:-1, 1:-1, :] = b[:,:,:]
    # plt.imshow(a.astype(np.int))

result = (a[:,::-1,:] + a)/4
result[:,:,0] = 1

import cv2
cv2.imwrite('stat_2d.png', result)