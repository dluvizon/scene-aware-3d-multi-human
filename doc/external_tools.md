## External Tools

### 1. Monocular Depth Estimation

We use a monocular depth estimation from [ [MiDaS/DPT](https://github.com/isl-org/DPT) ]. This is provided as a submodule in `tools/DPT`.

**1.1 Get the source code, apply modifications and download weights**

```bash
git submodule update --init --recursive

pushd tools/DPT
git apply  ../patches/midas_f43ef9e.patch
popd
```

Download the pretrained weights from [ [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) ] and copy the `.pt` file to `tools/DPT/weights/`.


**1.2 Test**

```bash
# This code should work with our standard environment
conda activate multi-human-mocap
pushd tools/DPT

inputpath="../../data/mupots-3d-eval/TS1/images"
outputpath="../../output/Test_TS1_DPT_large_monodepth"

python run_monodepth.py \
  --input_path $inputpath \
  --output_path $outputpath \
  --model_type dpt_large

popd
```
The predicted depth maps should appear in the output folder as `img_000000.png, ...`.


### 2. 2D Human Pose Estimation and Tracking

We use [ [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) ] for predicting 2D poses and tracking. This is provided as a submodule in `tools/AlphaPose`.

**2.1 Get the source code, apply modifications and download weights**

```bash
git submodule update --init --recursive

pushd tools/AlphaPose
git apply  ../patches/alphapose_d97acd0.patch
popd
```

- Download the pretrained [ [Fast Pose](https://github.com/MVIG-SJTU/AlphaPose/blob/d97acd01deba163855bde226f567022f96222138/docs/MODEL_ZOO.md#mscoco-dataset) ] model and copy it to `pretrained_models/fast_res50_256x192.pth`.
- Download the wrights for [ [YOLOv3](https://pjreddie.com/media/files/yolov3-spp.weights) ] detectors and copy the file to `detector/yolo/data/yolov3-spp.weights`.
- Download the [ [Human-ReID tracker](https://github.com/MVIG-SJTU/AlphaPose/tree/d97acd01deba163855bde226f567022f96222138/trackers#1-human-reid-based-tracking-recommended) ] weights and copy the file to `tools/AlphaPose/trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`.


**2.2 Install AlphaPose Conda environment**

Please follow the instructions to install a Conda environment from [ [here](https://github.com/MVIG-SJTU/AlphaPose/blob/d97acd01deba163855bde226f567022f96222138/docs/INSTALL.md#recommended-install-with-conda) ].

**Important:** keep a separated Conda environment `alphapose` for this tool.

**2.3 Test**

```bash
conda activate alphapose
pushd tools/AlphaPose

inputpath="../../data/mupots-3d-eval/TS1/images"
outputpath="../../output/Test_TS1_AlphaPose"

python3 scripts/demo_inference.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
  --checkpoint pretrained_models/fast_res50_256x192.pth \
  --indir ${inputpath} \
  --outdir ${outputpath} \
  --pose_track

popd
```
Predictions should be stored in `output/Test_TS1_AlphaPose/alphapose-results.json`.


## 3. Initial SMPL Parameters

We use [ [ROMP](https://github.com/Arthur151/ROMP) ] for SMPL parameters prediction. This is provided as a submodule in `tools/ROMP`.

**3.1 Get the source code, apply modifications and download weights**

```bash
git submodule update --init --recursive
conda activate multi-human-mocap

pushd tools/ROMP
git apply  ../patches/romp_f5b87be.patch

pushd simple_romp
python setup.py install
popd

popd
```

**3.2 Test**

```bash
# This code should work with our standard environment
conda activate multi-human-mocap
pushd tools/ROMP

inputpath="../../data/mupots-3d-eval/TS1/images"
outputpath="../../output/Test_TS1_ROMP"

romp --mode=video --calc_smpl \
    -i=${inputpath} \
    -o=${outputpath}

popd
```
Predictions should be stored in `output/Test_TS1_ROMP/img_000000.npz, ...`.

## 4. Instance Segmentation

We use [ [Mask2Former](https://github.com/facebookresearch/Mask2Former) ] for instance segmentation. This is provided as a submodule in `tools/Mask2Former`.

**4.1 Get the source code**

```bash
git submodule update --init --recursive

pushd tools/Mask2Former
git apply  ../patches/mask2former_16c3bee.patch
popd
```

Please follow the instructions to install a Conda environment from [ [here](https://github.com/facebookresearch/Mask2Former/blob/16c3beedc48fe40665253af01f580269b0e37711/INSTALL.md) ].

**Important:** keep a separated Conda environment `mask2former` for this tool.

**4.2 Test**

```bash
conda activate mask2former
pushd tools/Mask2Former

inputpath="../../data/mupots-3d-eval/TS1/images"
outputpath="../../output/Test_TS1_Mask2Former"

python run_instance_segmentation.py \
    --input ${inputpath} \
    --output ${outputpath}

popd
```
Predictions should be stored in `output/Test_TS1_Mask2Former/img_000000.png, ...`.