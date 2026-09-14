#!/bin/bash

# Distributed training configuration
master_addr=127.0.0.1
master_port=29500
nproc_per_node=1

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=$1
run_name=$2

# Dataset configuration
datasets=sportsqa_train

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=16
num_train_epochs=3
lora_rank=16
lora_alpha=32
lora_dropout=0.05

# Video configuration
video_fps=2
video_min_frames=8
video_max_frames=16
video_min_pixels=200704
video_max_pixels=802816

# Output configuration: replace this placeholder before running on the server.
output_root=PATH_TO_SPORTSQA_WORK_ROOT
output_dir=${output_root}/checkpoints/${run_name}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --data_packing False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --lora_enable True \
    --lora_r ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --model_max_length 4096 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --video_fps ${video_fps} \
    --video_min_frames ${video_min_frames} \
    --video_max_frames ${video_max_frames} \
    --video_min_pixels ${video_min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --gradient_checkpointing True \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --max_steps -1"

# Launch training
torchrun --nproc_per_node=${nproc_per_node} \
         --master_addr=${master_addr} \
         --master_port=${master_port} \
         ${entry_file} ${args}
