#!/bin/bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export SPORTSQA_EVAL_PROFILE=8f-634
export SPORTSQA_VIDEO_FRAMES=8
export SPORTSQA_VIDEO_MAX_PIXELS=401408

exec bash "${script_dir}/run_table10_qwen.sh" "$@"
