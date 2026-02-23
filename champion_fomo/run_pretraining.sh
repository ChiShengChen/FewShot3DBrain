#!/bin/bash

# Code
src_path='/home/jaume/Desktop/Code/fomo25/src'

# Data
data_path='/media/jaume/T7'
src_data=${data_path}/FOMO-MRI
pretrain_data=${src_data}/fomo-60k_baseline_preprocess/FOMO60k
save_dir_test=${src_data}/model_test_jaume_upload  # Directory to save pretraining outputs

# Options
model_name="mmunetvae"
epochs=100
warmup_epochs=5
mask_patch_size=4
mask_ratio=0.2
patch_size=64
batch_size=4
learning_rate=1e-4
num_devices=1
num_workers=2
checkpoint_every_n_epochs=25
optimizer="AdamW"
augmentation_preset="none"
experiment="test_experiment_upload"  # Weights & Biases experiment name

# == Pretraining ==
python ${src_path}/pretrain.py \
--save_dir ${save_dir_test} \
--pretrain_data_dir ${pretrain_data} \
--model_name ${model_name} \
--epochs ${epochs} \
--warmup_epochs ${warmup_epochs} \
--mask_patch_size ${mask_patch_size} \
--mask_ratio ${mask_ratio} \
--patch_size ${patch_size} \
--batch_size ${batch_size} \
--learning_rate ${learning_rate} \
--num_devices ${num_devices} \
--num_workers ${num_workers} \
--optimizer ${optimizer} \
--augmentation_preset ${augmentation_preset} \
--checkpoint_every_n_epochs ${checkpoint_every_n_epochs} \
--experiment ${experiment}