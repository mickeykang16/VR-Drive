#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:nuscenes-devkit/python-sdk


CONFIG=$1
GPUS=$3
PORT=${PORT:-29210}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=3 python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG --launcher pytorch ${@:3}


#  python3 tools/test_pkl.py projects/configs/diffusiondrive_configs/ours_novel_view_depth1.py iter_66000.pth --result_file work_dirs/ours_novel_view_depth1/results_orig.pkl --eval bbox --det_thresh 0.1 