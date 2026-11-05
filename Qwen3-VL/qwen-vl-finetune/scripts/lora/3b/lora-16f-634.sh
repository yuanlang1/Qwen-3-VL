#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/lora-common.sh" 16 401408 lora-16f-634 "$@"
