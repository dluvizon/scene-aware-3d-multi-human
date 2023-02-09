#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [clip path]"
    exit
fi
CLIP_PATH=`realpath $1`

source  ~/.bashrc # or replace it by any other script that initializes

##############################################################################
### Predict Depth Maps
##############################################################################
conda activate multi-human-mocap
pushd ./tools/DPT
python run_monodepth.py \
    --input_path "${CLIP_PATH}/images" \
    --output_path "${CLIP_PATH}/DPT_large_monodepth" \
    --model_type dpt_large
conda deactivate
popd

##############################################################################
### Predict AlphaPose
##############################################################################
conda activate alphapose
pushd ./tools/AlphaPose
python3 scripts/demo_inference.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --indir "${CLIP_PATH}/images" \
    --save_img \
    --outdir "${CLIP_PATH}/AlphaPose" \
    --pose_track
conda deactivate
popd

##############################################################################
### Predict ROMP
##############################################################################
conda activate multi-human-mocap
pushd ./tools/ROMP
romp --mode=video --calc_smpl \
    -i="${CLIP_PATH}/images" \
    -o="${CLIP_PATH}/ROMP_Predictions"
conda deactivate
popd

##############################################################################
### Predict Segmentation
##############################################################################
conda activate mask2former
pushd ./tools/Mask2Former
python run_instance_segmentation.py \
    --input "${CLIP_PATH}/images" \
    --output "${CLIP_PATH}/Mask2Former_Instances"
conda deactivate
popd
