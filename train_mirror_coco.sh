CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_mirror_coco.py \
--backbone vitb_hybrid \
--session_name mirror_coco_001 \
--lr 0.05 \
--batch_size 1 \
--crop_size 384 \
-g 4 \
--max_epoches 5