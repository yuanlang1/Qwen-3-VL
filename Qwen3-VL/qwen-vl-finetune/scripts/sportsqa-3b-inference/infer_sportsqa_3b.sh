#!/bin/bash

set -euo pipefail

# This runner writes only Sports-QA predictions. It never invokes exact scoring,
# the LLM judge, or reads SILICONFLOW_API_KEY.
export USE_TF=0
export USE_TORCH=1

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/sportsqa-3b-inference/infer_sportsqa_3b.sh MODEL RUN_NAME [ADAPTER_PATH]" >&2
    exit 2
fi

model=$1
run_name=$2
adapter_path=${3:-}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)

# Server-local Sports-QA paths.
prepared_root=/home/user/data/sportsqa_prepared
video_root=/home/user/data/export_test
output_root=/home/user/data/sportsqa_output/qwen2.5-vl

split=test

# GPU generation parameters. batch_size does not change DataLoader task size.
# Start conservatively in a memory-limited Pod; increase only after a full run.
batch_size=2
max_new_tokens=32

# Video sampling parameters.
video_frames=${SPORTSQA_VIDEO_FRAMES:-8}
video_min_pixels=${SPORTSQA_VIDEO_MIN_PIXELS:-50176}
video_max_pixels=${SPORTSQA_VIDEO_MAX_PIXELS:-200704}
prompt_style=yang-0s
profile_suffix=${SPORTSQA_EVAL_PROFILE:+_${SPORTSQA_EVAL_PROFILE}}

# CPU DataLoader/decoder parameters. Each prefetched item is one decoded video group.
# Switch to torchcodec after it is installed and verified on the server.
video_backend=torchvision
decoder_threads=2
num_workers=1
prefetch_factor=1
persistent_workers=false

# Main-process processor/H2D parameters.
use_fast_processor=true
pin_memory=false
timeout=120
multiprocessing_context=spawn

for required_path in "${model}" "${prepared_root}" "${video_root}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

prediction_file=${output_root}/${run_name}/predictions/${split}_${prompt_style}${profile_suffix}.jsonl

infer_args=(
    --model-name-or-path "${model}"
    --manifest "${prepared_root}/${split}_manifest.json"
    --video-root "${video_root}"
    --output-file "${prediction_file}"
    --resume
    --batch-size "${batch_size}"
    --max-new-tokens "${max_new_tokens}"
    --temperature 0
    --video-frames "${video_frames}"
    --video-min-pixels "${video_min_pixels}"
    --video-max-pixels "${video_max_pixels}"
    --video-backend "${video_backend}"
    --decoder-threads "${decoder_threads}"
    --prompt-style "${prompt_style}"
    --num-workers "${num_workers}"
    --prefetch-factor "${prefetch_factor}"
    --timeout "${timeout}"
    --multiprocessing-context "${multiprocessing_context}"
)

if [ -n "${adapter_path}" ]; then
    infer_args+=(--adapter-path "${adapter_path}")
fi
if [ "${persistent_workers}" = true ]; then
    infer_args+=(--persistent-workers)
fi
if [ "${pin_memory}" = false ]; then
    infer_args+=(--no-pin-memory)
fi
if [ "${use_fast_processor}" = false ]; then
    infer_args+=(--no-use-fast-processor)
fi

echo "Inference only: ${prediction_file}"
python "${project_root}/tools/infer_sportsqa.py" "${infer_args[@]}"
