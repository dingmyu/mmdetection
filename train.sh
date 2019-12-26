#!/usr/bin/env bash
set -x

# single machine
# srun -p ad_lidar --gres=gpu:4 -n1 --kill-on-bad-exit=1 ./tools/dist_train.sh configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py 4 --validate

# multi machines
srun -p ad_lidar --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py --launcher="slurm" --validate
