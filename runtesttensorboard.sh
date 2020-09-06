#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5

source /mnt/data/Rohit/VideoCapsNet/env/attention_ocr/bin/activate

tensorboard --logdir=logs/test/cat/ --host=localhost --port=6008
