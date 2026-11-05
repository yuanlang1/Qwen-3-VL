# Sports-QA server workflow

This workflow uses the official Sports-QA `train.json`, `val.json`, `test.json`,
and `ans2cls.json`. Dataset and work paths are explicit placeholders rather than
environment variables. The optional LLM-judge workflow reads only its API key from
the process environment.

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

### Validate videos before training

Before an SFT run, create an auditable filtered annotation. The validator always
checks required fields and video-file existence; `--verify-video` additionally
uses `ffprobe` to require a readable video stream. It does not change the source
annotation.

```bash
python tools/validate_sportsqa.py \
  --annotation /data/sportsqa/prepared/train.json \
  --video-root /data/sportsqa/videos \
  --output-dir /data/sportsqa/validation \
  --verify-video
```

This writes `valid_train.json`, `quarantined_train.jsonl`, and
`validation_summary_train.json`. Point `SPORTSQA_TRAIN["annotation_path"]` to
the valid annotation only after reviewing the quarantine file. Add
`--fail-on-invalid` in CI when any quarantined sample should block the run.

During training, a sample that still cannot be read is retried three times and
then raises an error with its dataset index, `qa_id`, video path, and root cause.
It is never replaced with a neighbouring QA record.

## 3. Validation-only zero-shot baselines

First edit `video_frames`, `video_min_pixels`, and `video_max_pixels` at the
top of `scripts/sportsqa_eval.sh`. Keep the same setting for all three models,
and evaluate on `val` only while choosing it. Start the smoke test with
`video_frames=8`, `video_min_pixels=50176`, and `video_max_pixels=200704`.

```bash
bash scripts/sportsqa_eval.sh Qwen/Qwen2.5-VL-3B-Instruct sportsqa-qwen25vl-3b
bash scripts/sportsqa_eval.sh Qwen/Qwen3-VL-4B-Instruct sportsqa-qwen3vl-4b
bash scripts/sportsqa_eval.sh Qwen/Qwen2.5-VL-7B-Instruct sportsqa-qwen25vl-7b
```

For a server smoke test, change `sample_num=0` to a small positive number in
the same script. Select one frame configuration by validation Macro-F1.

The evaluator uses a DataLoader inference pipeline by default. Consecutive QAs for
the same video form one CPU task: the video is decoded once, then expanded into its
ordered QA samples. The main process runs the processor for each QA and sends up to
`batch_size` samples to `model.generate()`. Therefore `batch_size` controls only the
GPU batch and does not multiply the amount of work assigned to each CPU worker. There
is no visual-feature cache, so the model still encodes the video once per QA.

The parameters are separated by stage:

| Stage | Parameters | Meaning |
| --- | --- | --- |
| GPU generation | `batch_size`, `max_new_tokens` | Number of decoded QAs per model call and output length |
| CPU decode/DataLoader | `video_backend`, `decoder_threads`, `num_workers`, `prefetch_factor`, `persistent_workers`, `timeout`, `multiprocessing_context` | Decoder implementation, CPU concurrency, and number of decoded video groups queued per worker |
| Main-process processor/H2D | `use_fast_processor`, `pin_memory` | Processor implementation and pinned-memory transfer preparation |
| Video sampling | `video_frames`, `video_min_pixels`, `video_max_pixels` | Frame count and visual token budget |

The default `video_backend=torchcodec` is strict: startup fails before model
inference if TorchCodec cannot be imported, rather than silently selecting
torchvision. This repository pins `torchcodec==0.7.0` for `torch==2.8.0`; the
server must also expose shared FFmpeg 4-9 libraries. Check the server environment
with:

```bash
python -c "import torch, torchcodec; from torchcodec.decoders import VideoDecoder; print(torch.__version__, torchcodec.__version__)"
ffmpeg -version
```

Use `video_backend=decord` only after installing Decord, or
`video_backend=torchvision` when its slower CPU path is intentional. Set
`video_backend=auto` only if fallback is desired. Keep
`num_workers * decoder_threads` below the CPU core budget. Tune `num_workers`
first, then `prefetch_factor`, and only increase `batch_size` for separate GPU
throughput experiments. The processor uses `do_resize=False` because
`qwen-vl-utils` already resizes decoded frames, so regenerate older baselines before
comparing predictions. The pipeline remains compatible with `--resume`.

## 4. LoRA fine-tuning and checkpoint selection

### Fixed-frame Qwen2.5-VL-3B profiles

The launchers in `scripts/lora/3b/` share `lora-common.sh`. Configure its model
and output paths on the server, and configure the dataset registry as above.
The existing `sportsqa_sft.sh` entry point is unchanged.

| Launcher | Fixed frames | Per-frame min pixels | Per-frame max pixels |
| --- | ---: | ---: | ---: |
| `lora-8f-448.sh` | 8 | 50176 | 200704 |
| `lora-8f-634.sh` | 8 | 50176 | 401408 |
| `lora-16f-448.sh` | 16 | 50176 | 200704 |
| `lora-16f-634.sh` | 16 | 50176 | 401408 |

`448` and `634` label pixel-area budgets, not forced square dimensions. Each
profile has its own checkpoint directory under `checkpoints/<profile-name>`.
Training uses `video_nframes` to pass `nframes` to `qwen-vl-utils`, exactly as
the inference pipeline does. The processor does not sample or resize again,
and no FPS is passed to the processor. Temporal positions therefore use the
processor's default FPS (2.0 in Transformers 4.57.0), rather than the video's
effective sampled FPS. The existing inference evaluator still supplies sampled
FPS; its frame/pixel budget matches, but its temporal scale can differ.
Videos with fewer original frames than requested fail explicitly.

The profiles retain LoRA r=16/alpha=32/dropout=0.05, learning rate 1e-5,
effective batch size 16 on one GPU, and 3 epochs. Their sequence limit is 8192
to accommodate the 16-frame high-pixel profile plus text. Samples exceeding
the limit or without answer labels fail instead of being silently truncated.

```bash
bash scripts/lora/3b/lora-8f-448.sh
bash scripts/lora/3b/lora-8f-634.sh
bash scripts/lora/3b/lora-16f-448.sh
bash scripts/lora/3b/lora-16f-634.sh
```

Use the same launcher with `--eval` to pass its exact video budget to the
existing deterministic evaluator. Configure that evaluator's prepared/video/
output paths on the server, including videos for the selected split.

```bash
bash scripts/lora/3b/lora-16f-634.sh --eval \
  /data/sportsqa/output/checkpoints/lora-16f-634/checkpoint-1000 val
```

The split defaults to `val`; use `test` only after checkpoint selection.
This option runs exact scoring, not the separate LLM judge. If using the
judge workflow, select its matching frame/pixel profile and the same
`video_min_pixels=50176` explicitly.

### Existing multi-model launchers

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

## 6. Table 10-style Qwen evaluation with an LLM judge

The isolated Yang-style Qwen workflow is in
[`tools/sportsqa-qwen-llm-judge`](tools/sportsqa-qwen-llm-judge/README.md), with its
entry point at `scripts/sft-3b-llm-judge/run_table10_qwen.sh`. It fixes the split to
`test`, uses the original question as the zero-shot prompt, preserves the existing
exact-match report, and writes a separate semantic report judged by SiliconFlow
DeepSeek-V3. The semantic score is protocol-comparable with Yang et al. but is not an
exact historical reproduction because Yang et al. used GPT-4 as the judge.
