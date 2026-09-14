# Sports-QA server workflow

This workflow uses the official Sports-QA `train.json`, `val.json`, `test.json`,
and `ans2cls.json`. It does not read `.env` files or shell environment variables.

## 1. Replace the path placeholders

All paths are intentionally explicit placeholders. Replace them with absolute
server paths before execution.

| File | Replace |
| --- | --- |
| `qwenvl/data/__init__.py` | The three `PATH_TO_SPORTSQA_*_ANNOTATION` values and `PATH_TO_SPORTSQA_VIDEO_ROOT` |
| `scripts/sportsqa_sft.sh` | `PATH_TO_SPORTSQA_WORK_ROOT` |
| `scripts/sportsqa_eval.sh` | `PATH_TO_SPORTSQA_PREPARED_ROOT`, `PATH_TO_SPORTSQA_VIDEO_ROOT`, and `PATH_TO_SPORTSQA_WORK_ROOT` |

For example, after preparing to `/data/sportsqa/prepared`, configure the dataset
registry as follows:

```python
SPORTSQA_TRAIN = {
    "annotation_path": "/data/sportsqa/prepared/train.json",
    "data_path": "/data/sportsqa/videos",
}
```

Use the matching `val.json` and `test.json` paths for the other two entries.

## 2. Prepare training JSON and evaluation manifests

Run this from the `qwen-vl-finetune` directory. The command neither downloads
nor copies videos.

```bash
python tools/prepare_sportsqa.py \
  --metadata-root /data/sportsqa/meta-data \
  --video-root /data/sportsqa/videos \
  --output-dir /data/sportsqa/prepared
```

It creates `train.json`, `val.json`, and `test.json` for SFT, independent
`*_manifest.json` files for evaluation, and `answer_to_id.json` for scoring.
For a metadata-only smoke test, append `--allow-missing --limit-per-split 2`.

If downloaded video filenames do not follow metadata IDs, pass
`--video-index /data/sportsqa/video_index.json`. The index maps a metadata video
ID to a path relative to `/data/sportsqa/videos`.

## 3. Validation-only zero-shot baselines

First edit the frame settings at the top of `scripts/sportsqa_eval.sh`. Keep the
same setting for all three models, and evaluate on `val` only while choosing it.

```bash
bash scripts/sportsqa_eval.sh Qwen/Qwen2.5-VL-3B-Instruct sportsqa-qwen25vl-3b
bash scripts/sportsqa_eval.sh Qwen/Qwen3-VL-4B-Instruct sportsqa-qwen3vl-4b
bash scripts/sportsqa_eval.sh Qwen/Qwen2.5-VL-7B-Instruct sportsqa-qwen25vl-7b
```

For a server smoke test, change `max_samples=0` to a small positive number in
the same script. Select one frame configuration by validation Macro-F1.

## 4. LoRA fine-tuning and checkpoint selection

Edit the training values at the top of `scripts/sportsqa_sft.sh`, then launch
the model-specific wrapper:

```bash
bash scripts/sportsqa_qwen25vl_3b.sh
bash scripts/sportsqa_qwen3vl_4b.sh
bash scripts/sportsqa_qwen25vl_7b.sh
```

The defaults are LoRA `r=16`, `alpha=32`, batch size 1, gradient accumulation
16, and 3 epochs. Evaluate candidate checkpoints on `val` only:

```bash
bash scripts/sportsqa_eval.sh \
  Qwen/Qwen3-VL-4B-Instruct sportsqa-qwen3vl-4b-step1000 \
  /data/sportsqa/output/checkpoints/sportsqa-qwen3vl-4b/checkpoint-1000
```

## 5. One final test evaluation

After choosing the checkpoint and frame count on validation, retain those
settings and provide `test` as the fourth argument:

```bash
bash scripts/sportsqa_eval.sh \
  Qwen/Qwen3-VL-4B-Instruct sportsqa-qwen3vl-4b-final \
  /data/sportsqa/output/checkpoints/sportsqa-qwen3vl-4b/checkpoint-1000 test
```

The metrics JSON contains overall Accuracy and Macro-F1 plus question-type and
sport-category groups. The details JSONL preserves raw and normalized answers
for error analysis.
