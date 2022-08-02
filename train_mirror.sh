

# CUDA_VISIBLE_DEVICES=3,5,6,7 \
# python train_mirror.py \
# --backbone vitb_hybrid \
# --session_name mirror_023 \
# --lr 0.05 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --batch_size 1 \
# --crop_size 384 \
# -g 4 \
# --max_epoches 15


# infer getam
CUDA_VISIBLE_DEVICES=6,7 \
python infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_023_last.pth \
--LISTpath voc12/train.txt \
--heatmap output/getam \
--address 9999 \
--out_cam output/cam_npy/ \