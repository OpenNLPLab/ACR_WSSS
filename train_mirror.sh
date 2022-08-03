

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python train_mirror.py \
# --backbone vitb_hybrid \
# --session_name mirror_025 \
# --lr 0.05 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --batch_size 1 \
# --crop_size 384 \
# -g 4 \
# --max_epoches 15


# infer getam
CUDA_VISIBLE_DEVICES=0,1 \
python infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_023_last.pth \
--LISTpath voc12/train.txt \
--heatmap output/getam \
--address 9999 \
--out_cam output/cam_npy/ \
--start_layer 9 \