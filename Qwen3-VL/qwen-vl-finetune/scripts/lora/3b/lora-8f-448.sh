#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/lora-common.sh" 8 200704 lora-8f-448 "$@"
