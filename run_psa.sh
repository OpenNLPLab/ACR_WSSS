cd /home/users/u5876230/mirror/

CUDA_VISIBLE_DEVICES=0,1 \
python infer_getam.py \
--backbone vitb_hybrid \
--weights weight/mirror_023_last.pth \
--LISTpath voc12/train_aug.txt  \
--heatmap output/getam \
--address 9999 \
--low_alpha 1 \
--high_alpha 12 \
--out_cam /data/u5876230/mirror_output/psa_cam/ \
--start_layer 10 \
--out_crf /data/u5876230/mirror_output/psa_crf \
--voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/ \


# cd /home/users/u5876230/MCTformer/

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python psa/train_aff.py --weights /home/users/u5876230/wsss/res38_cls.pth \
                        --voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012 \
                        --la_crf_dir /data/u5876230/mirror_output/psa_crf_1 \
                        --ha_crf_dir /data/u5876230/mirror_output/psa_crf_12 \


python psa/infer_aff.py --weights resnet38_aff.pth \
                    --infer_list voc12/train_id.txt \
                    --cam_dir /data/u5876230/mirror_output/psa_cam/ \
                    --voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/ \
                    --out_rw /data/u5876230/mirror_output/psa_pseudo/ \

python evaluation.py --list voc12/train_id.txt \
                     --gt_dir /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClass/ \
                     --logfile evallog.txt \
                     --type png \
                     --predict_dir /data/u5876230/mirror_output/psa_pseudo/ \
                     --comment "train 1464"

