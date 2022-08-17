# infer getam
CUDA_VISIBLE_DEVICES=7 \
python3 infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_029_last.pth \
--LISTpath voc12/train.txt \
--heatmap output/heatmap \
--address 9999 \
--IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
--start_layer 10 \
--out_cam output/cam_npy/ \
--getam_func grad 
# --out_ha_crf output/getam/ \
# --out_la_crf output/getam \
# --low_alpha 1 \
# --high_alpha 18 \


