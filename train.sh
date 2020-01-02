#!/usr/bin/env bash
set -x

# single machine
# srun -p ad_lidar --gres=gpu:4 -n1 --kill-on-bad-exit=1 ./tools/dist_train.sh configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py 4 --validate

# multi machines
# srun -p ad_lidar --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py --launcher="slurm" --validate


#srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate


#srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/kitti_fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate
srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/kitti_in_coco_fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate
