# ----------------------------------------- MMDET & PYCHARM -----------------------------------------------------------

# git Fork
# pycharm VCS checkout
# pycharm Deployment

# INSTALL miniconda3
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# gcc -v
# check that gcc version is not 4.8.5, use other versions (4.9.4, > 4 is better, 5.4.0) in /mnt/lustre/share/gcc/
# export PATH=/mnt/lustre/share/gcc/gcc-5.4/bin/:$PATH

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install cython
pip install -r requirements.txt  # (include mmcv)

# rm -rf build & python setup.py develop  # After build, u can see mmdet/version.py

mkdir data
cd data
ln -s /mnt/lustre/share/DSK/datasets/mscoco2017 coco

wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth


# -------------------------------------------- TRAIN ------------------------------------------------------------------

# single machine
srun -p ad_lidar --gres=gpu:4 -n1 --kill-on-bad-exit=1 ./tools/dist_train.sh configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py 4 --validate

# multi machines
srun -p ad_lidar --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py --launcher="slurm" --validate
srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate

# -------------------------------------------- TEST -------------------------------------------------------------------

# single GPU
srun -p ad_lidar --gres=gpu:1 -n1 --kill-on-bad-exit=1 python tools/test.py configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py \
  work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/epoch_24.pth --out work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/result.pkl  --eval bbox

# multi GPUs, can not use port
srun -p ad_lidar --gres=gpu:4 -n1 --kill-on-bad-exit=1 ./tools/dist_test.sh configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py \
  work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/latest.pth 4 --out work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/result_multi.pkl --eval bbox

# multi GPUs with port
srun -p ad_lidar --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/test.py \
  configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/latest.pth \
  --out work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/result.pkl --launcher="slurm"

# ---------------------------------------- KITTI DATASET --------------------------------------------------------------
### ONLY SUPPORT DIST TRAIN

# KITTI_PATH: /mnt/lustre/dingmingyu/Research/M3D-RPN/data/kitti
# TRAIN_FILE: kitti_tools/split1/train.txt
# VAL_FILE: kitti_tools/split1/val.txt

ln -s KITTI_PATH data/kitti
python kitti_tools/split1/setup_val.py
sh kitti_tools/devkit/cpp/build.sh

python kitti_tools/split1/convert_datasets/kitti_in_coco.py
python kitti_tools/split1/convert_datasets/kitti.py
python kitti_tools/split1/convert_datasets/kitti_test.py

# if use split2, change the script path in eval_hooks.py
# script = os.path.join(os.getcwd(), 'kitti_tools', 'split1', 'devkit', 'cpp', 'evaluate_object')
