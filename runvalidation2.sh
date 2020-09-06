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
  CUDA_VISIBLE_DEVICES=3 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/val_tf_records/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=84 --split=test --eval_log_dir=logs/val/icdarcomp15/ --batch_size=1 |& tee -a log_1aicdarcomp15
  sleep 20

  CUDA_VISIBLE_DEVICES=3 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/3_aICDAR19VideosTrainnewcpy/tfval/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=2091 --split=validation --eval_log_dir=logs/val/icdar19/ --batch_size=1 |& tee -a log_3aicdar19
  sleep 20

  CUDA_VISIBLE_DEVICES=3 python -u eval.py --dataset_dir=/mnt/data/Rohit/ACMData/1a_CATVideosTrain/tftest/ --train_log_dir=logs/train/ --number_of_steps=1 --num_batches=8978 --split=validation --eval_log_dir=logs/test/cat/ --batch_size=1 |& tee -a log_1acat_test
  sleep 20
done
