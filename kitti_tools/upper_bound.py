import easydict as edict
import subprocess
import os
import sys
import re
import math
import numpy as np

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
            if label_type in TYPE2LABEL:
                all_boxes.append((label_type, x, y, x2, y2, h3d, w3d, l3d, cx3d, cy3d, cz3d, rotY))
    return all_boxes


def read_kitti_pre(file):
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

    pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
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
            prob = float(parsed.group(16))
            # all_boxes.append(((x+x2)/2, (y+y2)/2, width, height))
            if label_type in TYPE2LABEL:
                all_boxes.append((label_type, x, y, x2, y2, h3d, w3d, l3d, cx3d, cy3d, cz3d, rotY, prob))
    return all_boxes

def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积

    iou = area / (carea + garea - area)

    return iou

os.chdir(sys.path[0])
os.system('mkdir test_new')
os.system('mkdir test_new/data')


for index in range(3769):
    if index % 200 == 0:
        print(index)
    #     if index >= 100:
    #         break
    id = '%06d' % index
    labelname = os.path.join('/mnt/lustre/dingmingyu/2020/mmdetection/kitti_tools/split1/validation/label_2',
                             id + '.txt')
    gts = read_kitti_label(labelname)
    labelname = os.path.join('./data', id + '.txt')
    pres = read_kitti_pre(labelname)

    fw = open('test_new/data/%s.txt' % id, 'w')
    for pre in pres:
        pre = list(pre)
        flag = 0
        for gt in gts:
            gt = list(gt)
            if pre[0] == gt[0]:
                iou = calculateIoU(gt[1:], pre[1:])
                if iou > 0.5:
                    #                     print(pre)
                    pre[11] = gt[11]  # Ry
                    #                     pre[5] = gt[5]  # h
                    #                     pre[6] = gt[6]  # w
                    #                     pre[7] = gt[7]  # l
                    #                     pre[8] = gt[8]  # x
                    #                     pre[9] = gt[9]  # y
                    pre[10] = gt[10]  # z
                    pre = [str(item) for item in pre]
                    pre.insert(1, '0')
                    pre.insert(1, '-1')
                    pre.insert(1, '-1')
                    print(' '.join(pre), file=fw)
                    flag = 1
                    break
        if flag == 0:
            pre = [str(item) for item in pre]
            pre.insert(1, '0')
            pre.insert(1, '-1')
            pre.insert(1, '-1')
            print(' '.join(pre), file=fw)
    fw.close()



script = os.path.join(
    '/mnt/lustre/dingmingyu/2020/mmdetection',
    'kitti_tools',
    'split1',
    'devkit',
    'cpp',
    'evaluate_object')
os.chdir('/mnt/lustre/dingmingyu/2020/mmdetection/')
print(os.path.join(os.getcwd()))
with open(os.devnull, 'w') as devnull:
    out = subprocess.check_output([script, os.path.join(sys.path[0], 'test_new')], stderr=devnull)
os.chdir(sys.path[0])
print(os.path.join(sys.path[0], 'test_new'))


# os.chdir('/mnt/lustre/dingmingyu/2020/mmdetection/')
import os.path as osp
results_path = osp.join(sys.path[0], 'test_new', 'data')



def parse_kitti_result(respath, mode='new'):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall(r'([\d]+\.?[\d]*)', line)

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


for lbl in ['Car', 'Cyclist', 'Pedestrian']:

    lbl = lbl.lower()

    respath_2d = os.path.join(results_path.replace(
        '/data', ''), 'stats_{}_detection.txt'.format(lbl))
    respath_gr = os.path.join(
        results_path.replace(
            '/data',
            ''),
        'stats_{}_detection_ground.txt'.format(lbl))
    respath_3d = os.path.join(
        results_path.replace(
            '/data',
            ''),
        'stats_{}_detection_3d.txt'.format(lbl))
#     print(respath_2d)
    if os.path.exists(respath_2d):
#         print(respath_2d)
        easy, mod, hard = parse_kitti_result(respath_2d, mode='old')

        print_str = 'R11_test_epoch {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)
        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_2d)

        print_str = 'R40_test_epoch {} 2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)
        print(print_str)

    if os.path.exists(respath_gr):
        easy, mod, hard = parse_kitti_result(respath_gr, mode='old')

        print_str = 'R11_test_epoch {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)

        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_gr)

        print_str = 'R40_test_epoch {} gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)

        print(print_str)

    if os.path.exists(respath_3d):
        easy, mod, hard = parse_kitti_result(respath_3d, mode='old')

        print_str = 'R11_test_epoch {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)

        print(print_str)

        easy, mod, hard = parse_kitti_result(respath_3d)

        print_str = 'R40_test_epoch {} 3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(
            1, lbl, easy, mod, hard)

        print(print_str)
# os.chdir(sys.path[0])