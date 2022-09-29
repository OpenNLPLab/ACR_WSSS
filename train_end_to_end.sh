CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_from_init_coco.py \
--backbone vitb_hybrid \
--session_name train_from_init_coco_001 \
--gpus 4 \
--nr 0 \
--max_epoches 20  \
--lr 0.02 \
--cls_step 0.5 \
--seg_lr_scale 0.1 \
--batch_size 2 \
--sal_loss True 