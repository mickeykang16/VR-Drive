# VR-Drive Environment Setting
```bash
# From the DiffusionDriveForward directory

conda create -n vrdrive python=3.8 -y
conda activate vrdrive

# (Optional) Check CUDA version (expect 11.6)
nvcc -V

python -m pip install --upgrade pip
python -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements_final.txt

# Build ops
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../../

# Install gaussian rasterization submodule
cd models/gaussian/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
cd ../../..

# Install mmcv / mmdetection
pip install mmcv-full==1.5.0

cd mmdetection
pip install -v -e .
cd ..

# PerceptualSimilarity
cd PerceptualSimilarity
python setup.py develop
cd ..

# nuScenes devkit
export PYTHONPATH="$PYTHONPATH:$(pwd)/nuscenes-devkit/python-sdk"

echo "✅ NOW READY TO RUN YOUR CODE"

```


# Dataset Preparation


```bash
export WORK_DIR="vrdrive"
export GPUS=4
export CONFIG="./projects/configs/vrdrive_configs/vrdrive.py"
python -m torch.distributed.run \
    --nproc_per_node=${GPUS} \
    --master_port=2333 \
    tools/train.py ${CONFIG} \
    --launcher pytorch \
    --deterministic \
    --work-dir ${WORK_DIR} \
    --no-validate
```

## 3. Evaluation

```bash
bash ./tools/dist_test.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    ckpt/diffusiondrive_stage2.pth \
    8 \
    --deterministic \
    --eval bbox
```

