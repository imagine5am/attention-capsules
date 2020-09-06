#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin/:$PATH
export CUDNN_PATH=/usr/local/cuda-8.0/lib64/libcudnn.so.5

source /mnt/data/Rohit/VideoCapsNet/env/attention_ocr/bin/activate

while true; do
  CUDA_VISIBLE_DEVICES=2 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/1a_CATVideosTrain/tftest/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=5825 --split=test --eval_log_dir=logs/test/cat/ |& tee -a log_1acat_test
  sleep 20
done