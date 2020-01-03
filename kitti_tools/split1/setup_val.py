# Author: Mingyu Ding
# Time: 2/1/2020 9:09 PM
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

import numpy as np
import sys
import os
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.
    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)


# base paths
base_data = os.path.join(os.getcwd(), 'data')

kitti_raw = dict()
kitti_raw['base'] = os.path.join(base_data, 'kitti')
kitti_raw['lab'] = os.path.join(kitti_raw['base'], 'training', 'label_2')

kitti_val = dict()
kitti_val['lab'] = os.path.join(os.getcwd(), 'kitti_tools', 'split1', 'validation', 'label_2')

val_file = 'kitti_tools/split1/val.txt'

# mkdirs
mkdir_if_missing(kitti_val['lab'])

print('Linking val')
text_file = open(val_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_val['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(kitti_raw['lab'], str(id) + '.txt'),
                       os.path.join(kitti_val['lab'], str(new_id) + '.txt'))

        imind += 1

text_file.close()

print('Done')
