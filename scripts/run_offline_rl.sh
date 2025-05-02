#!/bin/bash

set -x
set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_API_KEY=19063ca51ad554f5d50bce4a84d9aa6d36d720f9
export WANDB_PROJECT=offline_rl

model_path=/liziniu/models/RLHFlow-LLaMA3-SFT-v2

num_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $num_gpus GPUs"

seed=42
global_batch_size=128
pert_device_batch_size=4
gradient_accumulation_steps=$((global_batch_size / pert_device_batch_size / num_gpus))

learning_rate=5e-6
kl_coeff=0.1

time_stamp=$(date +%Y-%m-%d-%H-%M-%S)
train_tokenized_file=./data/ultrafeedback_4k_tokenized.jsonl
output_dir=./outputs/offline_rl_rlhflow_llama3_sft_v2_ultrafeedback_4k_lr_${learning_rate}_kl_${kl_coeff}_${time_stamp}
mkdir -p $output_dir

deepspeed train.py \
    --deepspeed scripts/zero3.json \
    --seed $seed \
    --model_name_or_path $model_path \
    --train_tokenized_file $train_tokenized_file \
    --output_dir $output_dir \
    --per_device_train_batch_size $pert_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate $learning_rate \
    --lr_scheduler_type cosine \
    --kl_coeff $kl_coeff \
    --save_only_model True \
    --remove_unused_columns False \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --logging_steps 5 \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
   2>&1 | tee $output_dir/training.log