# infer getam
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 infer_getam_irn.py \
# --backbone vitb_hybrid \
# --weights weight/mirror_029_last.pth \
# --LISTpath voc12/train_aug.txt \
# --address 5463 \
# --IMpath /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages \
# --irn_out_cam /data/u5876230/mirror_output/irn_cam_npy \
# --start_layer 10 \
# --getam_func grad \
# --heatmap output/irn_heatmap \



# fg_list=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
# bg_list=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)


# a=10
# b=20

# val=`expr $a \* $b`
# echo "a * b : $val"

# c=$(awk 'BEGIN{print 7.01*5-4.01 }')
# echo $c

# cd /home/users/u5876230/irn/


# for CONF_FG in 0.35
# do 
#     for CONF_BG in 0.1
#     do 
#         if  (( $(awk 'BEGIN {print ("'$CONF_BG'" >= "'$CONF_FG'")}') ))
#         then
#             echo "skip!"
#             echo 'fg' $CONF_FG
#             echo 'bg' $CONF_BG
#         else
#             echo 'go!'
#             echo 'fg' $CONF_FG
#             echo 'bg' $CONF_BG
#             CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sample.py \
#             --conf_fg_thres $CONF_FG --conf_bg_thres $CONF_BG \
#             # --train_cam_pass False \
#             # --make_cam_pass False \
#             # --eval_cam_pass False \
#             # --cam_to_ir_label_pass True \
#             # --train_irn_pass True \
#             # --make_ins_seg_pass False \
#             # --eval_ins_seg_pass False \
#             # --make_sem_seg_pass True \
#             # --eval_sem_seg_pass True \ 
#         fi
# done
# done


