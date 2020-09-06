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

while true; do
  CUDA_VISIBLE_DEVICES=1 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/1a_CATVideosTrain/tfval/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=3832 --split=validation --eval_log_dir=logs/val/cat/ --batch_size=1 |& tee -a log_1acat
  sleep 20

  CUDA_VISIBLE_DEVICES=1 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/crop_val_tf_records/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=29 --split=test --eval_log_dir=logs/val/icdarcomp15crop/ --batch_size=1 |& tee -a log_1aicdarcomp15crop
  sleep 20

  CUDA_VISIBLE_DEVICES=1 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/5aSynthVideosE2E/tfval/  --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=312 --split=test --eval_log_dir=logs/val/synth/ --batch_size=1 |& tee -a log_5asynth
  sleep 20
done
