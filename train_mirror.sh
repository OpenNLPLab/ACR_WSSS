

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
CUDA_VISIBLE_DEVICES=0 \
python3 infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_023_last.pth \
--LISTpath voc12/train.txt \
--heatmap output/getam \
--address 9999 \
--out_cam output/cam_npy/ \
--IMpath /home/SENSETIME/sunweixuan/pascal/JPEGImages \
--out_la_crf output/getam/ \
--out_ha_crf output/getam/ \


# infer getam 2
# CUDA_VISIBLE_DEVICES=0 \
# python3 infer_getam_2.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_023_last.pth \
# --LISTpath voc12/train.txt \
# --heatmap output/getam \
# --address 9999 \
# --IMpath /home/SENSETIME/sunweixuan/pascal/JPEGImages \
# --out_cam output/cam_npy/ \
# --out_la_crf output/getam/ \
# --out_ha_crf output/getam/ \