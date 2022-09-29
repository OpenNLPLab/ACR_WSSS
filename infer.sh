CAM_NPY_DIR=/home/users/u5876230/mirror/output/cam_npy/
# CAM_NPY_DIR=/data/u5876230/mirror/output/cam_npy/

echo $CAM_NPY_DIR

# infer getam
CUDA_VISIBLE_DEVICES=1 \
python3 infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_029_last.pth \
--LISTpath voc12/train.txt \
--heatmap output/heatmap \
--address 2345 \
--IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
--start_layer 10 \
--out_cam $CAM_NPY_DIR \
--getam_func cam_grad_s \
--aff True
# --pseudo output/pseudo \
# --out_ha_crf output/getam/ \
# --out_la_crf output/getam \
# --low_alpha 1 \
# --high_alpha 18 \


# CUDA_VISIBLE_DEVICES=6 \
# python3 infer_getam_2.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_029_last.pth \
# --LISTpath voc12/train.txt \
# --address 8888 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --start_layer 10 \
# --getam_func grad \
# --out_ha_crf output/getam/ \
# --out_la_crf output/getam \
# --low_alpha 1 \
# --high_alpha 18 \


# python3 infer_getam_irn.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_029_last.pth \
# --LISTpath voc12/train_aug.txt \
# --heatmap output/heatmap \
# --address 9999 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --irn_out_cam /data/u5876230/mirror_output/irn_cam_npy \

cd /home/users/u5876230/MCTformer/

python evaluation.py \
--list voc12/train_id.txt \
--gt_dir /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClassAug \
--logfile /data/MCTformer_results/MCTformer_v2/attn-patchrefine-npy/evallog.txt \
--type npy \
--curve True \
--predict_dir $CAM_NPY_DIR \
--comment haha 