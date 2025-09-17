L=[1.0,1.0,0.8]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:36684 --resume    \
--config-file ./configs/tf_learning_merge/updown/updown.yaml     \
INFERENCE.STYLE 1  \
DATALOADER.ANNO_WEIGHT_TRAIN $L \
OUTPUT_DIR ./work_dirs/v1
