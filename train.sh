#!/usr/bin/env bash
set -x

card=8
percard=8
srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate