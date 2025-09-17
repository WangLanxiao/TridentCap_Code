CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:21084 --resume \
--config-file ./configs/tf_learning_merge/updown/updown_rl.yaml \
DATALOADER.ANNO_FOLDER_TRAIN ./open_source_dataset/stylecaption_merge/train.pkl \
DATALOADER.ANNO_FOLDER_VAL ./open_source_dataset/stylecaption_merge/val.pkl \
DATALOADER.ANNO_FOLDER_TEST ./open_source_dataset/stylecaption_merge/test.pkl \
SCORER.GT_PATH ./open_source_dataset/stylecaption_merge/train_gts.pkl \
SCORER.CIDER_CACHED ./open_source_dataset/stylecaption_merge/train_cider.pkl \
MODEL.WEIGHTS  ./work_dirs/V1/model.pth   \
OUTPUT_DIR ./work_dirs/v1_rl


