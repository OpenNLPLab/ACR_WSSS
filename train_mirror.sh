CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_mirror.py \
--backbone vitb_hybrid \
--session_name mirror_035 \
--lr 0.01 \
--IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
--batch_size 1 \
--crop_size 384 \
-g 4 \
--max_epoches 15 \
--address 8888



# infer getam
# CUDA_VISIBLE_DEVICES=7 \
# python3 infer_getam.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_031_last.pth \
# --LISTpath voc12/val.txt \
# --heatmap output/heatmap \
# --address 9999 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --start_layer 10 \
# --out_cam output/cam_npy/ \
# --out_ha_crf output/getam/ \
# --out_la_crf output/getam \
# --low_alpha 1 \
# --high_alpha 18 \


# CUDA_VISIBLE_DEVICES=0 \
# python3 infer_getam_irn.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_029_last.pth \
# --LISTpath voc12/train_aug.txt \
# --heatmap output/heatmap \
# --address 9999 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --irn_out_cam /data/u5876230/mirror_output/irn_cam_npy \