#!/bin/bash
# export CUDNN_PATH=/mnt/data/cuda/lib64/libcudnn.so.7
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-10.0/bin/:$PATH
# export LD_LIBRARY_PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$LD_LIBRARY_PATH
# export PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5

source ../../env/attention_ocr/bin/activate
tensorboard --logdir=logs/train/ --host=localhost --port=6006

