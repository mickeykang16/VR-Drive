#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:nuscenes-devkit/python-sdk

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# 🔥 10000~60000 범위에서 랜덤 포트 자동 선택
PORT=$((10000 + RANDOM % 50000))

echo "Using random PORT: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=3 python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
