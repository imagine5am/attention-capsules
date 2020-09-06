#!/bin/bash
# export CUDNN_PATH=/mnt/data/cuda/lib64/libcudnn.so.7
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-10.0/bin/:$PATH
# export LD_LIBRARY_PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$LD_LIBRARY_PATH
# export PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5

source /mnt/data/Rohit/VideoCapsNet/env/attention_ocr/bin/activate

tensorboard --logdir=logs/val/cat/ --host=localhost --port=6007
#tensorboard --logdir=logs/val/icdarcomp15crop/ --host=localhost --port=6007
#tensorboard --logdir=logs/val/synth/ --host=localhost --port=6007
#tensorboard --logdir=logs/val/icdarcomp15/ --host=localhost --port=6007
#tensorboard --logdir=logs/val/icdar19/ --host=localhost --port=6007
