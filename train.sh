#!/usr/bin/env bash
set -x

srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate
#srun -p ad_lidar --gres=gpu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_CocoFormat.py --launcher="slurm" --validate


