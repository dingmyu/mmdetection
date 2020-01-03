#!/usr/bin/env bash
set -x

srun -p ad_lidar --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/test.py \
  configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/latest.pth \
  --out work_dirs_kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/result.pkl --launcher="slurm"