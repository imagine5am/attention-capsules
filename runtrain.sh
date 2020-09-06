#!/bin/bash
#export CUDNN_PATH=/mnt/data/cuda/lib64/libcudnn.so.7
#export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-10.0/bin/:$PATH
#export LD_LIBRARY_PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$LD_LIBRARY_PATH
#export PATH=/home/ganesh/rohitmntdata/MayurMTP/cuda/lib64/:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5
#source /mnt/data/Rohit/SynthVideosCapsnet/synthvid-caps27/bin/activate

source /mnt/data/Rohit/VideoCapsNet/env/attention_ocr/bin/activate

CUDA_VISIBLE_DEVICES=0 python train.py --train_log_dir=logs/train/ --caps_checkpoint=/home/ganesh/rohitmntdata/Attention_OCR/model_7.ckpt-83 &&

while true; do
  CUDA_VISIBLE_DEVICES=0 python train.py --train_log_dir=logs/train/ &&
  sleep 200
done

