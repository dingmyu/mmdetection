#!/usr/bin/env bash

# single GPU
#srun -p ad_rd --gres=gpu:1 -n1 --kill-on-bad-exit=1 python tools/test.py configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py \
#  work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/epoch_24.pth --out work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/result.pkl  --eval bbox

# multi GPUs
srun -p ad_rd --gres=gpu:8 -n1 --kill-on-bad-exit=1 ./tools/dist_test.sh configs/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py \
  work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/epoch_24.pth 4 --out work_dirs/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu/result_multi.pkl --eval bbox