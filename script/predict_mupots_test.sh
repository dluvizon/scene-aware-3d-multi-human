#!/usr/bin/env bash

export QT_QPA_PLATFORM=offscreen

source  ~/.bashrc # or replace it by any other script that initializes
conda activate multi-human-mocap

sel_sets=(1)
for ts in ${sel_sets[@]}
do
  python -m mhmocap.predict_mupots \
    --configs_yml configs/predict_mupots.yml \
    --ts_id $ts \
    --num_iter 100 \
    --output_path="./output/mupots-Test"
done
