#!/bin/bash

set -e

# Sports-QA inference uses PyTorch only. Avoid importing a TensorFlow build that
# may be incompatible with the server image's NumPy/CUDA runtime.
export USE_TF=0
export USE_TORCH=1

# Sports-QA paths: replace these placeholders before running on the server.
prepared_root=/home/user/data/sportsqa_prepared
video_root=/home/user/data/export_test
output_root=/home/user/data/sportsqa_output/qwen2.5-vl
answer_map=${prepared_root}/answer_to_id.json

# GPU generation parameters. batch_size does not change DataLoader task size.
batch_size=1
max_new_tokens=32

# Video sampling parameters
video_frames=8
video_min_pixels=50176
video_max_pixels=200704
max_samples=0

# CPU DataLoader/decoder parameters. Each prefetched item is one decoded QA.
video_backend=torchcodec
decoder_threads=2
num_workers=4
prefetch_factor=2
persistent_workers=false

# Main-process processor/H2D parameters
use_fast_processor=true
pin_memory=true
timeout=120
multiprocessing_context=spawn

# Model configuration
llm=$1
run_name=$2
adapter_path=""
split=val

if [ "$#" -ge 3 ]; then
    adapter_path=$3
fi
if [ "$#" -ge 4 ]; then
    split=$4
fi

# Output configuration
prediction_file=${output_root}/predictions/${run_name}_${split}.jsonl
metric_file=${output_root}/metrics/${run_name}_${split}.json
details_file=${output_root}/metrics/${run_name}_${split}_details.jsonl

# Inference arguments
infer_args="
    --model-name-or-path ${llm} \
    --manifest ${prepared_root}/${split}_manifest.json \
    --video-root ${video_root} \
    --output-file ${prediction_file} \
    --batch-size ${batch_size} \
    --max-new-tokens ${max_new_tokens} \
    --temperature 0 \
    --video-frames ${video_frames} \
    --video-min-pixels ${video_min_pixels} \
    --video-max-pixels ${video_max_pixels} \
    --video-backend ${video_backend} \
    --decoder-threads ${decoder_threads} \
    --num-workers ${num_workers} \
    --prefetch-factor ${prefetch_factor} \
    --timeout ${timeout} \
    --multiprocessing-context ${multiprocessing_context}"

if [ -n "${adapter_path}" ]; then
    infer_args="${infer_args} --adapter-path ${adapter_path}"
fi
if [ "${max_samples}" -gt 0 ]; then
    infer_args="${infer_args} --limit ${max_samples}"
fi
if [ "${persistent_workers}" = true ]; then
    infer_args="${infer_args} --persistent-workers"
fi
if [ "${pin_memory}" = false ]; then
    infer_args="${infer_args} --no-pin-memory"
fi
if [ "${use_fast_processor}" = false ]; then
    infer_args="${infer_args} --no-use-fast-processor"
fi

python tools/infer_sportsqa.py ${infer_args}

# Metric arguments
eval_args="
    --manifest ${prepared_root}/${split}_manifest.json \
    --predictions ${prediction_file} \
    --answer-map ${answer_map} \
    --output-file ${metric_file} \
    --details-file ${details_file}"

if [ "${max_samples}" -gt 0 ]; then
    eval_args="${eval_args} --limit ${max_samples}"
fi

python tools/eval_sportsqa.py ${eval_args}
