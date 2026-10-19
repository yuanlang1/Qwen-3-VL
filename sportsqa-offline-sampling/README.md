# SportsQA offline non-uniform sampling

This standalone tool migrates Hour-LLaVA's query-aware Sports-TCS frame-selection data preparation. It does not modify `Qwen3-VL`, invoke training, or change any inference-time video loader.

For each Sports-QA QA record it produces seven visual retrieval queries, CLIP scores over adaptive-FPS candidate frames, and an ordered non-uniform selection. The default output is 16 frames per QA: 12 relevance-focused Slow frames and 4 context-preserving Fast frames. Slow and Fast are sampling-origin labels only.

## Install

Use an environment with a PyTorch build compatible with the intended GPU, then install this directory:

```powershell
cd "E:\codex\qwen rp\sportsqa-offline-sampling"
pip install -r requirements.txt
pip install -e .
```

`queries` reads the query-service key only from `DASHSCOPE_API_KEY`. It never writes the key to output files. `scores` requires an OpenCLIP-compatible CUDA environment by default; use `--device cpu` only for debugging.

## Stages

All commands require the official metadata directory, extracted-video root, and a new output directory. Use an output directory dedicated to one preprocessing configuration.

```powershell
$env:DASHSCOPE_API_KEY = "..."

python -m sportsqa_offline_sampling queries `
  --metadata-root "E:\learn\recurrence\Hour-LLaVA\hourllava\data\meta-data" `
  --video-root "D:\sportsqa-videos" `
  --output-dir "D:\sportsqa-offline-v1" `
  --splits train val test

python -m sportsqa_offline_sampling scores `
  --metadata-root "E:\learn\recurrence\Hour-LLaVA\hourllava\data\meta-data" `
  --video-root "D:\sportsqa-videos" `
  --output-dir "D:\sportsqa-offline-v1" `
  --splits train val test

python -m sportsqa_offline_sampling select `
  --metadata-root "E:\learn\recurrence\Hour-LLaVA\hourllava\data\meta-data" `
  --video-root "D:\sportsqa-videos" `
  --output-dir "D:\sportsqa-offline-v1" `
  --splits train val test `
  --frame-budget 16 --num-queries 7

python -m sportsqa_offline_sampling validate `
  --metadata-root "E:\learn\recurrence\Hour-LLaVA\hourllava\data\meta-data" `
  --video-root "D:\sportsqa-videos" `
  --output-dir "D:\sportsqa-offline-v1" `
  --splits train val test `
  --frame-budget 16 --num-queries 7 --check-videos
```

For a 100-QA smoke test, add `--limit 100` to every command and use a separate output directory. `all` executes the same stages in order:

```powershell
python -m sportsqa_offline_sampling all --metadata-root "..." --video-root "..." --output-dir "..." --limit 100
```

The exit code is nonzero if any records fail or validation finds incomplete output. Failures are written under `failures/`; successful records remain usable when the command is rerun.

## Sampling policy

The first stage decodes no more than four uniformly spaced previews at up to 224×224. Qwen3-VL-Flash infers an optional broad sport type solely from visible frames and emits seven ranked retrieval queries. Answer text, answer classes, question types, and path-derived sport labels are not supplied to the query generator, scorer, or selector.

Candidate frames use 4 FPS for videos up to 30 seconds, 2 FPS through 120 seconds, and 1 FPS thereafter; the final frame is always included. OpenCLIP ViT-L/14 with OpenAI weights scores each frame against all seven queries. The selector averages query scores, applies Gaussian smoothing (radius 4, sigma 1), detects clips with `mean + 0.5 × standard deviation`, adapts the threshold where a pool is too small, and then samples 75% Slow and 25% Fast frames. All emitted indices are sorted, unique, zero-based original decoder frame indices.

## Outputs and resume behavior

```
OUTPUT_DIR/
  queries/{split}.jsonl
  scores/{split}.jsonl
  cache/features/{CLIP-configuration-fingerprint}/*.npz
  selections/{split}/q7_f16.jsonl
  reports/validation_{split}_q7_f16.json
  failures/*.jsonl
```

The score stage caches candidate-frame CLIP features by video and configuration. The selection records include `split`, `qa_id`, original and resolved video paths, question, source frame count/FPS, final indices, decoder-origin timestamps, Slow/Fast labels, clip IDs, relevance scores, and the effective threshold.

Every stage writes a configuration fingerprint. A changed metadata, query, CLIP, or selection configuration fails before artifacts are mixed. Use a new output directory for a new experiment. Interrupted final JSONL lines are discarded safely on resume; malformed records in the middle of a JSONL file cause a clear failure.

Changing `--frame-budget` or `--num-queries` only requires re-running `select` and `validate`, because query banks and CLIP scores are preserved. This first version saves frame references only; later Qwen integration must read these exact original-frame indices and timestamps instead of returning to uniform video sampling.
