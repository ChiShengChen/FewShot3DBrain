#!/bin/bash

# Code
src_path='/home/jaume/Desktop/Code/fomo25/src'
# src_path='/cluster/home/ja1659/Code/fomo25/src'

# Data
data_path='/media/jaume/T71'
# data_path='/data/bdip2/jbanusco/fomo25'

src_data=${data_path}/FOMO-MRI
save_dir_test=${src_data}/model_test_jaume_upload  # Directory to save pretraining outputs  model_test_jaume_6/model_test_jaume_upload

# Finetuning data
fine_tune_data=${data_path}/finetuning_data_preprocess/mimic-pretreaining-preprocessing

# == Finetuning ==
model_name="mmunetvae"
# pretrained_model=${save_dir_test}/models/FOMO60k/${model_name}/versions/version_0/epoch=99.ckpt
pretrained_model=${save_dir_test}/models/FOMO60k/${model_name}/versions/version_0/last.ckpt

# Shared Options
epochs=100
patch_size=64
batch_size=4
learning_rate=1e-4
num_devices=1
num_workers=1
split_method="kfold"
split_param=5 # Number of folds

# === Function to run finetuning for a specific task and fold ===
run_finetuning() {
    local task_id=$1
    local task_epochs=$2
    local task_experiment=$3
    local augmentation=$4
    local path_splits=$5
    local fold_idx=$6

    echo "----------------------------------------------------------------"
    echo "Starting: Task ${task_id} | Fold ${fold_idx}/${split_param} | Epochs ${task_epochs}"
    echo "Experiment: ${task_experiment}"
    echo "----------------------------------------------------------------"

    python "${src_path}/finetune.py" \
        --save_dir "${save_dir_test}" \
        --data_dir "${fine_tune_data}" \
        --model_name "${model_name}" \
        --epochs "${task_epochs}" \
        --patch_size "${patch_size}" \
        --batch_size "${batch_size}" \
        --learning_rate "${learning_rate}" \
        --num_devices "${num_devices}" \
        --num_workers "${num_workers}" \
        --augmentation_preset "${augmentation}" \
        --experiment "${task_experiment}" \
        --pretrained_weights_path "${pretrained_model}" \
        --split_idx "${fold_idx}" \
        --split_param "${split_param}" \
        --split_method "${split_method}" \
        --path_splits "${path_splits}" \
        --taskid "${task_id}"
}

#====== Task 1
epochs=100
augmentation_preset="none"
experiment="test_experiment_upload_task1"  # Weights & Biases experiment name    
path_splits=${data_path}/data/splits_final/task1/splits_final_no_test.json

# Run only fold 0
# run_finetuning 1 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} 0

# Run all folds
# for i in {0..4}; do
#     run_finetuning 1 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} $i
# done

#====== Task 2
epochs=100
augmentation_preset="none"
experiment="test_experiment_upload_task2"  # Weights & Biases experiment name
path_splits=${data_path}/data/splits_final/task2/nnunet_experiments/splits_final_no_test.json

# Run only fold 0
run_finetuning 2 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} 0

# Run all folds
# for i in {0..4}; do
#     run_finetuning 2 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} $i
# done

#====== Task 3
epochs=100
augmentation_preset="none"
experiment="test_experiment_upload_task3"  # Weights & Biases experiment name
spath_splits=${data_path}/data/splits_final/task3/splits_final_no_test.json

# Run only fold 0
# run_finetuning 3 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} 0

# Run all folds
# for i in {0..4}; do
#     run_finetuning 3 ${epochs} ${experiment} ${augmentation_preset} ${path_splits} $i
# done