#!/bin/bash

set -euo pipefail

# Sports-QA inference uses PyTorch only. Avoid importing a TensorFlow build that
# may be incompatible with the server image's NumPy/CUDA runtime.
export USE_TF=0
export USE_TORCH=1

if [ "${1:-}" = "--judge-only" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: bash scripts/sft-3b-llm-judge/run_table10_qwen.sh --judge-only RUN_NAME" >&2
        exit 2
    fi
    judge_only=true
    model=
    run_name=$2
    adapter_path=
elif [ "$#" -ge 2 ] && [ "$#" -le 3 ]; then
    judge_only=false
    model=$1
    run_name=$2
    adapter_path=${3:-}
else
    echo "Usage: bash scripts/sft-3b-llm-judge/run_table10_qwen.sh MODEL RUN_NAME [ADAPTER_PATH]" >&2
    echo "   or: bash scripts/sft-3b-llm-judge/run_table10_qwen.sh --judge-only RUN_NAME" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
judge_dir=${project_root}/tools/sportsqa-qwen-llm-judge

# Replace this placeholder only in the server-local copy. Do not commit a real key.
export SILICONFLOW_API_KEY='YOUR_SILICONFLOW_API_KEY'
if [[ "${SILICONFLOW_API_KEY}" == 'YOUR_SILICONFLOW_API_KEY' ]]; then
    echo "Replace SILICONFLOW_API_KEY in ${script_dir}/run_table10_qwen.sh." >&2
    exit 2
fi

# Replace these explicit placeholders before running on the server.
prepared_root=PATH_TO_SPORTSQA_PREPARED_ROOT
video_root=PATH_TO_SPORTSQA_VIDEO_ROOT
output_root=PATH_TO_SPORTSQA_WORK_ROOT

required_paths=("${prepared_root}" "${output_root}")
if [ "${judge_only}" = false ]; then
    required_paths+=("${video_root}")
fi
for configured_path in "${required_paths[@]}"; do
    if [[ "${configured_path}" == PATH_TO_* ]]; then
        echo "Replace Sports-QA path placeholders in ${script_dir}/run_table10_qwen.sh." >&2
        exit 2
    fi
done

# Yang-style Qwen inference profile. The paper does not disclose video sampling;
# freeze this documented profile rather than selecting it on the test set.
split=test
batch_size=1
max_new_tokens=32
video_frames=8
video_min_pixels=50176
video_max_pixels=200704
prompt_style=yang-0s
max_samples=0
cache_video_features=false

# SiliconFlow semantic judge. The API key is read only by the judge from
# SILICONFLOW_API_KEY and is never passed on the command line or written to disk.
judge_model=deepseek-ai/DeepSeek-V3
judge_batch_size=100
judge_max_tokens=4096
judge_timeout_seconds=240
judge_max_in_flight_batches=4
# SiliconFlow L0 account ceilings. The judge uses 80% of these ceilings so
# unrelated calls on the same account retain headroom.
judge_rate_limit_rpm=1000
judge_rate_limit_tpm=50000
judge_rate_limit_safety_factor=0.8
judge_max_attempts=8
judge_retry_backoff_seconds=15
judge_max_retry_backoff_seconds=300

run_root=${output_root}/${run_name}
prediction_file=${run_root}/predictions/${split}_yang-0s.jsonl
exact_metric_file=${run_root}/metrics/${split}_yang-0s_exact.json
exact_details_file=${run_root}/metrics/${split}_yang-0s_exact_details.jsonl
judge_label=hybrid-rule-exact-deepseek-v3-compact-no-thinking-batch${judge_batch_size}
judgment_file=${run_root}/judgments/${split}_yang-0s_${judge_label}.jsonl
semantic_metric_file=${run_root}/metrics/${split}_yang-0s_${judge_label}.json
manifest=${prepared_root}/${split}_manifest.json

if [ "${judge_only}" = false ]; then
    infer_args=(
        --model-name-or-path "${model}"
        --manifest "${manifest}"
        --video-root "${video_root}"
        --output-file "${prediction_file}"
        --resume
        --batch-size "${batch_size}"
        --max-new-tokens "${max_new_tokens}"
        --temperature 0
        --video-frames "${video_frames}"
        --video-min-pixels "${video_min_pixels}"
        --video-max-pixels "${video_max_pixels}"
        --prompt-style "${prompt_style}"
    )
    if [ -n "${adapter_path}" ]; then
        infer_args+=(--adapter-path "${adapter_path}")
    fi
    if [ "${max_samples}" -gt 0 ]; then
        infer_args+=(--limit "${max_samples}")
    fi
    if [ "${cache_video_features}" = true ]; then
        infer_args+=(--cache-video-features)
    fi

    python "${project_root}/tools/infer_sportsqa.py" "${infer_args[@]}"
elif [ ! -f "${prediction_file}" ]; then
    echo "Cannot start judge-only mode: missing predictions at ${prediction_file}." >&2
    exit 2
fi

exact_args=(
    --manifest "${manifest}"
    --predictions "${prediction_file}"
    --answer-map "${prepared_root}/answer_to_id.json"
    --output-file "${exact_metric_file}"
    --details-file "${exact_details_file}"
)
if [ "${max_samples}" -gt 0 ]; then
    exact_args+=(--limit "${max_samples}")
fi
python "${project_root}/tools/eval_sportsqa.py" "${exact_args[@]}" > /dev/null

judge_args=(
    --manifest "${manifest}"
    --predictions "${prediction_file}"
    --output-file "${judgment_file}"
    --rule-details "${exact_details_file}"
    --model "${judge_model}"
    --temperature 0
    --max-tokens "${judge_max_tokens}"
    --batch-size "${judge_batch_size}"
    --timeout-seconds "${judge_timeout_seconds}"
    --max-in-flight-batches "${judge_max_in_flight_batches}"
    --rate-limit-rpm "${judge_rate_limit_rpm}"
    --rate-limit-tpm "${judge_rate_limit_tpm}"
    --rate-limit-safety-factor "${judge_rate_limit_safety_factor}"
    --max-attempts "${judge_max_attempts}"
    --retry-backoff-seconds "${judge_retry_backoff_seconds}"
    --max-retry-backoff-seconds "${judge_max_retry_backoff_seconds}"
    --resume
)
if [ "${max_samples}" -gt 0 ]; then
    judge_args+=(--limit "${max_samples}")
fi
python "${judge_dir}/judge_sportsqa_semantic.py" "${judge_args[@]}"

summary_args=(
    --manifest "${manifest}"
    --judgments "${judgment_file}"
    --output-file "${semantic_metric_file}"
    --require-complete
)
if [ "${max_samples}" -gt 0 ]; then
    summary_args+=(--limit "${max_samples}")
fi
python "${judge_dir}/summarize_sportsqa_semantic.py" "${summary_args[@]}"
