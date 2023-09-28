CAM_NPY_DIR=/home/users/u5876230/ACR/mirror/output/cam_npy/
# CAM_NPY_DIR=/data/u5876230/mirror/output/cam_npy/

echo $CAM_NPY_DIR

# infer getam
# CUDA_VISIBLE_DEVICES=1 \
# python3 infer_getam.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_003_last.pth \
# --LISTpath voc12/train.txt \
# --heatmap output/heatmap \
# --address 2345 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --start_layer 10 \
# --out_cam $CAM_NPY_DIR \
# --getam_func grad \
# --aff False \




cd /home/users/u5876230/MCTformer/

python evaluation.py \
--list voc12/train_id.txt \
--gt_dir /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClassAug \
--logfile /data/MCTformer_results/MCTformer_v2/attn-patchrefine-npy/evallog.txt \
--type npy \
--curve True \
--predict_dir $CAM_NPY_DIR \
--comment haha 