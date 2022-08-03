CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_mirror_single_stage.py \
--backbone vitb_hybrid \
--session_name single_001 \
--lr 0.05 \
--IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
--batch_size 1 \
--crop_size 384 \
-g 4 \
--max_epoches 20 \