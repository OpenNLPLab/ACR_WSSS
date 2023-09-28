EXP_NAME=acr_coco_001

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_acr_coco.py \
--backbone vitb_hybrid \
--session_name $EXP_NAME \
--lr 0.05 \
--batch_size 1 \
--crop_size 384 \
-g 4 \
--max_epoches 5 \
--IMpath {coco train image path} \
--valpath {coco val image path} 