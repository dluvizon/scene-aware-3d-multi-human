#!/bin/bash -l

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [input_path] [output_path]"
    exit
fi
inputpath=`realpath $1`
outputpath=`realpath $2`

export QT_QPA_PLATFORM=offscreen

source  ~/.bashrc # or replace it by any other script that initializes
conda activate multi-human-mocap

python -m mhmocap.predict_internet \
  --configs_yml configs/default.yml \
  --input_path="${inputpath}" \
  --output_path="${outputpath}"

