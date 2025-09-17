#!/usr/bin/env bash

PTH_PATH="./work_dirs/v1"
subpth="xxxx.pth"
echo $PTH_PATH
for sub_style in  $(seq 1 4)
do
    if [ $t -gt -1 ];then
      echo ${subpth}
      echo ${sub_style}
      CUDA_VISIBLE_DEVICES=1 python ./train_net.py --num-gpus 1 \
      --config-file ./configs/tf_learning_merge/updown/updown.yaml \
      --eval-only \
      MODEL.WEIGHTS ${subpth} \
      DATALOADER.TEST_BATCH_SIZE 16 \
      DECODE_STRATEGY.BEAM_SIZE 3 \
      INFERENCE.STYLE ${sub_style} \
      OUTPUT_DIR ${PTH_PATH}
    fi
done
