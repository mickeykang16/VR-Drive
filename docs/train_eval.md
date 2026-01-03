# VR-Drive Training and Evaluation

## 1. Download the DiffusionDrive stage-2 ckpt from huggingface
https://huggingface.co/hustvl/DiffusionDrive/tree/main

```shell
ckpt
└── diffusiondrive_stage2.pth
```



## 2. Training


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

### Original View Evaluation
```bash
bash ./tools/dist_test.sh \
    projects/configs/vrdrive_configs/ours_original.py \
    work_space/vrdrive/checkpoint.pth \
    4 \
    --deterministic \
    --eval bbox
```


### Novel View Evaluation
```bash
bash ./tools/dist_test.sh \
    projects/configs/vrdrive_configs/ours_novel_view_***.py \
    work_space/vrdrive/checkpoint.pth \
    4 \
    --deterministic \
    --eval bbox
```

