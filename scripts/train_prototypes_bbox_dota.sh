#!/bin/bash

DATA_DIR=data
INIT_PROTOTYPES_PATH=run/init_prototypes_dior
backbone=dinov2

for dataset in dota
do
    for N in 5 
    do
        echo "Training box classifier model for the ${dataset} dataset using ${backbone} features with N=${N}"
        python train.py \
            --train_root_dir  /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train/images \
            --val_root_dir  /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train/images \
            --save_dir /data_student_2/zhouzhi/FSL_object_detection/ovdsat/run/train_dota/boxes/${dataset}_N${N} \
            --train_annotations_file /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train_dota_subset_N5.json\
            --val_annotations_file /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train_dota_finetune_val.json \
            --prototypes_path /data_student_2/zhouzhi/FSL_object_detection/ovdsat/run/init_prototypes_dota/boxes/dota_N5/prototypes_dinov2.pt \
            --backbone_type ${backbone} \
            --num_epochs 200 \
            --lr 2e-4 \
            --target_size 602 602 \
            --batch_size 8 \
            --num_neg 0 \
            --num_workers 2 \
            --iou_thr 0.1 \
            --conf_thres 0.2 \
            --scale_factor 1 \
            --annotations box \
            --only_train_prototypes
    done
done

