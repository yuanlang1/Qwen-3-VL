#!/bin/bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export SPORTSQA_EVAL_PROFILE=16f-634
export SPORTSQA_VIDEO_FRAMES=16
export SPORTSQA_VIDEO_MAX_PIXELS=401408
export SPORTSQA_BATCH_SIZE=6
sample_num=${SPORTSQA_SAMPLE_NUM:-0}
export SPORTSQA_SAMPLE_NUM=${sample_num}

exec bash "${script_dir}/infer_sportsqa_7b.sh" "$@"
