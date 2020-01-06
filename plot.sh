#!/usr/bin/env bash
set -x

dir=output/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/BASELINE_lr_0.002_nms_0.4_epoch_24_2020_01_06_15_35_58
python tools/analyze_logs.py plot_curve $dir/20200106_153600.log.json \
--keys loss_cls loss_bbox loss_centerness loss --out $dir/losses.pdf