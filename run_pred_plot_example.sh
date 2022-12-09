#!/bin/bash

# ckpt=../jenkins_experiments/0-kk/lightning_logs/version_15/checkpoints/epoch=2-step=21.ckpt
# data=../data/NPN_1155_part2.dat
# cfg=../configs/config_kk.yaml
# python pred_plot.py $cfg $ckpt $data

# exit 0

ckpt=../jenkins_experiments/0-kk-v/lightning_logs/version_49/checkpoints/epoch=99-step=700.ckpt
data=../data/NPN_1155_part2.dat
cfg=../configs/config_kk_v.yaml
python pred_plot.py $cfg $ckpt $data --feature-id 1

