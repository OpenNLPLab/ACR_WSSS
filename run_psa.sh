# for LOW_ALPHA in 2 4 6 8 10
# do 
#     for HIGH_ALPHA in 12 14 16 18 20 22 
#     do 

CAM_NPY_DIR=/data/u5876230/mirror_output/grad_aff_npy
BG_THR=0.55


LOW_ALPHA=4
HIGH_ALPHA=32
SESSION=grad_s_4_32
echo $SESSION
echo $LOW_ALPHA
echo $HIGH_ALPHA

cd /home/users/u5876230/mirror/

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python infer_getam.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_029_last.pth \
# --LISTpath voc12/train_aug.txt  \
# --heatmap output/getam_recam \
# --address 7777 \
# --low_alpha $LOW_ALPHA \
# --high_alpha $HIGH_ALPHA \
# --out_cam /data/u5876230/mirror_output/psa_cam/ \
# --start_layer 10 \
# --getam_func grad_s \
# --out_crf /data/u5876230/mirror_output/psa_crf \
# --voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/ \

# --out_la_crf output/test/ \
# --out_ha_crf output/test/ \

cd /home/users/u5876230/MCTformer/

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python psa/train_aff.py --weights /home/users/u5876230/wsss/res38_cls.pth \
#                         --voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012 \
#                         --la_crf_dir /data/u5876230/mirror_output/psa_crf_$LOW_ALPHA \
#                         --ha_crf_dir /data/u5876230/mirror_output/psa_crf_$HIGH_ALPHA \
#                         --session_name $SESSION \

echo /data/u5876230/mirror_output/$SESSION\_psa_pseudo_$BG_THR/
mkdir -p /data/u5876230/mirror_output/$SESSION\_psa_pseudo_$BG_THR/

cd /home/users/u5876230/MCTformer/
CUDA_VISIBLE_DEVICES=4,5,6,7 python psa/infer_aff.py --weights /home/users/u5876230/MCTformer/$SESSION.pth \
                    --infer_list psa/voc12/train_aug.txt \
                    --cam_dir $CAM_NPY_DIR \
                    --voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/ \
                    --out_rw /data/u5876230/mirror_output/$SESSION\_psa_pseudo_$BG_THR/ \
                    --bg_thr $BG_THR \
                    --beta 36 \

python evaluation.py --list voc12/train_id.txt \
                    --gt_dir /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClass/ \
                    --logfile evallog.txt \
                    --type png \
                    --predict_dir /data/u5876230/mirror_output/$SESSION\_psa_pseudo_$BG_THR/ \
                    --comment "train 1464"

# done
# done
