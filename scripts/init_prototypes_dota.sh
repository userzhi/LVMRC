backbone=dinov2
for dataset in dota
do
    for N in 5
    do
        echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

        python build_prototypes.py \
            --data_dir /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train/images \
            --save_dir /data_student_2/zhouzhi/FSL_object_detection/ovdsat/run/init_prototypes_dota/boxes/dota_N5 \
            --annotations_file /data_student_2/zhouzhi/FSL_object_detection/ovdsat/data/dota/train_dota_subset_N5.json \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --window_size 224 \
            --scale_factor 1 \
            --num_b 10 \
            --k 200 
    done
done


: set off=unix