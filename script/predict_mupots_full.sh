#!/bin/bash -l

export QT_QPA_PLATFORM=offscreen

source  ~/.bashrc # or replace it by any other script that initializes
conda activate multi-human-mocap

for ts in {1..20}
do
  python -m mhmocap.predict_mupots \
    --configs_yml configs/predict_mupots.yml \
    --ts_id $ts \
    --output_path="./output/mupots"
done
