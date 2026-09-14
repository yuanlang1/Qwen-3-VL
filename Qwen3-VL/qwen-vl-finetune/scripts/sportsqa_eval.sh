#!/bin/bash

set -e

# Sports-QA paths: replace these placeholders before running on the server.
prepared_root=PATH_TO_SPORTSQA_PREPARED_ROOT
video_root=PATH_TO_SPORTSQA_VIDEO_ROOT
output_root=PATH_TO_SPORTSQA_WORK_ROOT
answer_map=${prepared_root}/answer_to_id.json

# Evaluation hyperparameters
batch_size=1
max_new_tokens=32
video_frames=8
video_min_pixels=50176
video_max_pixels=200704
max_samples=0

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
    --video-max-pixels ${video_max_pixels}"

if [ -n "${adapter_path}" ]; then
    infer_args="${infer_args} --adapter-path ${adapter_path}"
fi
if [ "${max_samples}" -gt 0 ]; then
    infer_args="${infer_args} --limit ${max_samples}"
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
