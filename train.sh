#!/usr/bin/env bash
set -x
time=$(date "+%Y%m%d_%H%M%S")
card=4
percard=4
#srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/retinanet_free_anchor_x101-32x4d_fpn_1x.py --launcher="slurm" --validate
#srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py --launcher="slurm" --validate --timestamp $time
#srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/ga_faster_x101_32x4d_fpn_1x.py --launcher="slurm" --validate
#srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/faster_rcnn_x101_64x4d_fpn_1x.py --launcher="slurm" --validate
#srun -p ad_lidar --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/retinanet_x101_64x4d_fpn_1x.py --launcher="slurm" --validate

#srun -p ad_lidar -x SH-IDC1-10-5-36-213 --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/faster_rcnn_x101_64x4d_fpn_3d.py --launcher="slurm" --validate --timestamp $time

srun -p ad_lidar -x SH-IDC1-10-5-36-213 --gres=gpu:$percard --ntasks=$card --ntasks-per-node=$percard --cpus-per-task=5 --kill-on-bad-exit=1 python -u tools/train.py configs/kitti/fcos_x101_64x4d_gn_3d.py --launcher="slurm" --validate --timestamp $time
