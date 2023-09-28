EXP_NAME=acr_001

mkdir -p ./weight

cp train_mirror.sh weight/$EXP_NAME\.sh

# train acr
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_acr.py \
--backbone vitb_hybrid \
--session_name $EXP_NAME \
--lr 0.05 \
--IMpath {pascalvoc_img_path} \
--batch_size 1 \
--alpha 125 \
--crop_size 384 \
-g 4 \
--max_epoches 10 \
--address 2222 \

CAM_NPY_DIR={path_to_save_cam_npy}

echo $CAM_NPY_DIR

# infer cam
CUDA_VISIBLE_DEVICES=1 \
python3 infer_cam.py \
--backbone vitb_hybrid \
--weights weight/$EXP_NAME\_last.pth \
--LISTpath voc12/train.txt \
--heatmap {path to save cam heatmap} \
--address 2345 \
--IMpath {pascalvoc_img_path} \
--start_layer 10 \
--out_cam $CAM_NPY_DIR \
--getam_func grad \
--aff True

# evaluate cam
python evaluation.py \
--list voc12/train_id.txt \
--gt_dir {path to save pascal voc SegmentationClassAug} \
--logfile evallog.txt \
--type npy \
--curve True \
--predict_dir $CAM_NPY_DIR \
--comment haha 

echo $EXP_NAME