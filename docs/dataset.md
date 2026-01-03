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

Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```


Download the [NuScenes Novelview Testset](https://huggingface.co/datasets/mickeykang/VR-Drive)


Your nuScenes dataset should be organized according to the following structure:
```shell
data
└── nuscenes
    ├── can_bus
    ├── maps
    ├── samples
    ├── images_depth+1
    ├── images_height-0.7
    ├── images_height+1
    ├── images_pitch-10
    ├── images_pitch+5
    ├── sweeps
    ├── v1.0-test
    ├── v1.0-trainval
    └── val_token.txt
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```


### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```

