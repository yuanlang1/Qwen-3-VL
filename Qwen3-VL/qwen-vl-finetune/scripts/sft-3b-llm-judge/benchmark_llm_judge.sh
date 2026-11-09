#!/bin/bash
set -euo pipefail
echo "Legacy batch judge benchmark disabled after migration to single-QA v6." >&2
echo "Judge --batch-size is no longer supported. Use an eval profile runner or --judge-only." >&2
echo "Original script: /home/user/project/.judge-migrations/20261027T162618+0800-2764bc0/backup/scripts/sft-3b-llm-judge/benchmark_llm_judge.sh." >&2
exit 2
