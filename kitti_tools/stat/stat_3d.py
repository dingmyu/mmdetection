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
Aw = 123.62374/2  # 69.16813
Ah = 113.182/2  # 51.076797]

# trans_mean = [-0.00378265,  -0.00182043,  -0.00622482, 0.016801083, -0.03962014, 26.235588,  -0.04752721,  -0.02627299]
# trans_std = [0.08565227,   0.06028508,   0.11255758, 0.6852198, 0.38124686, 12.873396,  1.6905682,  1.7750752]
trans_mean = [-0.00378265,  -0.00182043,  -0.00622482, 0, 0, 26.235588,  -0.04752721,  -0.02627299]
trans_std = [0.08565227,   0.06028508,   0.11255758, 1, 1, 12.873396,  1.6905682,  1.7750752]

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
    x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
    y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
    z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

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
            obj.bbox_3d = [h3d, w3d, l3d, x_p, y_p, z_p, rotY, alpha]

            # for i in range(8):
            #     obj.bbox_3d[i] = (obj.bbox_3d[i] - mean_3d[i]) / std_3d[i]

            for i in range(3):
                obj.bbox_3d[i] = np.log(obj.bbox_3d[i]/mean_3d[i])

            for i in range(8):
                obj.bbox_3d[i] = (obj.bbox_3d[i] - trans_mean[i]) / trans_std[i]

            obj.bbox_2d = [x, y, x2, y2]
            obj.label = label_type

            # gts.append(obj)

            if label_type in TYPE2LABEL and occ <= 1 and not ign and height >= 23 and height <= 280:
                # print(obj.bbox_3d)
                bboxes.append(obj.bbox_2d)
                bboxes_3d.append(obj.bbox_3d)
                labels.append(TYPE2LABEL[label_type])
                # if obj.bbox_3d[3] > x2 or obj.bbox_3d[3] < x:
                #     print(obj.bbox_2d, obj.bbox_3d)
    return bboxes, bboxes_3d


import re
import math
import os
import numpy as np
all_boxes_2d, all_boxes_3d = [], []
for index in range(7481):
    if index % 500 == 0:
        print(index)
#     if index >= 100:
#         break
    id = '%06d' % index
    labelname = os.path.join('/mnt/lustre/dingmingyu/2020/mmdetection/', 'data', 'kitti', 'training', 'label_2', id + '.txt')
    calibname = labelname.replace('label_2', 'calib')
    calib = read_kitti_cal(calibname)
    label_2d, label_3d = read_kitti_label(labelname, calib, 1)
    all_boxes_2d.extend(label_2d)
    all_boxes_3d.extend(label_3d)


stat = np.zeros((375,1242,9), np.float)
for i in range(len(all_boxes_2d)):
    if i % 1000 == 0:
        print(i)
    x, y, x2, y2 = all_boxes_2d[i]
    h3d, w3d, l3d, x_p, y_p, z_p, rotY, alpha = all_boxes_3d[i]
    x, y = int(x), int(y)
    x2, y2 = math.ceil(x2), math.ceil(y2)
    width = x2 - x + 1
    height = y2 - y + 1
    h3d, w3d, l3d, z_p, rotY, alpha = [float(i) for i in [h3d, w3d, l3d, z_p, rotY, alpha]]
#     a[y:y2, x:x2, 0] += 1
    for i, ii in enumerate([1, width, height, h3d, w3d, l3d, z_p, rotY, alpha]):
        stat[y:y2, x:x2, i] += ii

a = stat[...].copy()
a[:,:,6][a[:,:,0]<=5] = 0 # z
a[:,:,5][a[:,:,0]<=5] = 0 # l
a[:,:,7][a[:,:,0]<=5] = 0 # rY
a[:,:,8][a[:,:,0]<=5] = 0 # alpha
a[:,:,0][a[:,:,0]<=5] = 0
a[:,:,0][a[:,:,0]==0] = 1
a[:,:,5:9] /= a[:,:,0:1]

a[:,:,0][a[:,:,6]==0] = 0
a[:,:,0][a[:,:,6]!=0] = 1


print((a[:,:,0]==0).sum()/(1242*375))

INF = 1e8
num = a[:,:,0]+a[:,::-1,0]
# flag = num.copy()
num[num==0]=1
import matplotlib.pyplot as plt
new_z = (a[:,:,6]+a[:,::-1,6])/num
new_z[new_z==0] = new_z.max()
new_h = (a[:,:,2] + a[:,::-1,2])/num
new_h[new_h==0] = INF
new_h[new_h==INF] = new_h.min()
new_h /= 113.182
new_w = (a[:,:,1] + a[:,::-1,1])/num
new_w[new_w==0] = INF
new_w[new_w==INF] = new_w.min()
new_w /= 123.62374
stat = np.zeros((375,1242,3), np.float)
stat[:,:,0] = new_z
stat[:,:,1] = new_w
stat[:,:,2] = new_h
np.save('stat.npy', stat)
# plt.imshow(new_w)
# plt.imshow(new_z)
# cv2.imwrite('stat_z.png', new_z)