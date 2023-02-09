#!/bin/bash -l

export QT_QPA_PLATFORM=offscreen

source  ~/.bashrc # or replace it by any other script that initializes
conda activate multi-human-mocap

python -m mhmocap.eval_mupots \
  --configs_yml configs/eval_mupots.yml \
  --input_path="./output/mupots"
