#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${project_root}"

# Profile parameters supplied by the four launchers.
video_nframes=$1
video_max_pixels=$2
run_name=$3
shift 3

# Model and paths: configure on the server.
# Dataset paths remain in qwenvl/data/__init__.py.
llm=Qwen/Qwen2.5-VL-3B-Instruct
output_root=PATH_TO_SPORTSQA_WORK_ROOT
output_dir=${output_root}/checkpoints/${run_name}
entry_file=qwenvl/train/train_qwen.py
deepspeed=./scripts/zero2.json

# Distributed training.
master_addr=127.0.0.1
master_port=29500
nproc_per_node=1

# Dataset and trainable components.
datasets=sportsqa_train
data_flatten=False
data_packing=False
tune_mm_vision=False
tune_mm_mlp=False
tune_mm_llm=False

# LoRA.
lora_enable=True
lora_rank=16
lora_alpha=32
lora_dropout=0.05

# Training hyperparameters.
num_train_epochs=3
batch_size=1
grad_accum_steps=16
learning_rate=1e-5
weight_decay=0
warmup_ratio=0.03
max_grad_norm=1
lr_scheduler_type=cosine
max_steps=-1

# Precision, sequence length, and memory.
bf16=True
model_max_length=8192
gradient_checkpointing=True

# Image pixels and per-frame video pixels. Fixed frame count comes from the profile.
max_pixels=50176
min_pixels=784
video_min_pixels=50176

# Evaluation, checkpoints, and logging.
eval_strategy=no
save_strategy=steps
save_steps=1000
save_total_limit=2
logging_steps=10
dataloader_num_workers=4

export USE_TF=0
export USE_TORCH=1

# Evaluate an adapter with exactly this profile's frame and per-frame pixel budget.
if [ "${1:-}" = "--eval" ]; then
    adapter_path=${2:?Usage: PROFILE.sh --eval ADAPTER_PATH [val|test]}
    split=${3:-val}
    exec bash scripts/sportsqa_eval.sh "${llm}" "${run_name}" \
        "${adapter_path}" "${split}" \
        "${video_nframes}" "${video_max_pixels}" "${video_min_pixels}"
fi

# Launch training: one argument per line, using the configuration above.
exec torchrun --nproc_per_node="${nproc_per_node}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    "${entry_file}" \
    --deepspeed "${deepspeed}" \
    --model_name_or_path "${llm}" \
    --dataset_use "${datasets}" \
    --data_flatten "${data_flatten}" \
    --data_packing "${data_packing}" \
    --tune_mm_vision "${tune_mm_vision}" \
    --tune_mm_mlp "${tune_mm_mlp}" \
    --tune_mm_llm "${tune_mm_llm}" \
    --lora_enable "${lora_enable}" \
    --lora_r "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
    --lora_dropout "${lora_dropout}" \
    --bf16 "${bf16}" \
    --output_dir "${output_dir}" \
    --num_train_epochs "${num_train_epochs}" \
    --per_device_train_batch_size "${batch_size}" \
    --gradient_accumulation_steps "${grad_accum_steps}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --warmup_ratio "${warmup_ratio}" \
    --max_grad_norm "${max_grad_norm}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --model_max_length "${model_max_length}" \
    --max_pixels "${max_pixels}" \
    --min_pixels "${min_pixels}" \
    --video_nframes "${video_nframes}" \
    --video_min_pixels "${video_min_pixels}" \
    --video_max_pixels "${video_max_pixels}" \
    --gradient_checkpointing "${gradient_checkpointing}" \
    --eval_strategy "${eval_strategy}" \
    --save_strategy "${save_strategy}" \
    --save_steps "${save_steps}" \
    --save_total_limit "${save_total_limit}" \
    --logging_steps "${logging_steps}" \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --run_name "${run_name}" \
    --max_steps "${max_steps}"
