# Sports-QA Qwen LLM Judge

This directory contains the isolated Table 10-style Qwen evaluation workflow:

1. Qwen video inference on the Sports-QA `test` split with the Yang zero-shot prompt.
2. Existing deterministic 191-class exact-match scoring for diagnostics.
3. Yang-style semantic scoring using SiliconFlow `deepseek-ai/DeepSeek-V3` as the judge.

The semantic score is comparable in protocol to Yang et al., but is **not** an exact
reproduction because Yang used GPT-4 as the judge. Keep the two metric files separate.

## Setup

Run from `qwen-vl-finetune`. Replace the three path placeholders and the Key placeholder
near the top of `scripts/sft-3b-llm-judge/run_table10_qwen.sh`:

```bash
export SILICONFLOW_API_KEY='YOUR_SILICONFLOW_API_KEY'
```

Replace the value only in the server-local script and do not commit a real key. The key
is never accepted as a command-line argument and is not stored in prediction, judgment,
or metric artifacts.

## Runs

Start with a small end-to-end smoke test by changing `max_samples` in
`scripts/sft-3b-llm-judge/run_table10_qwen.sh`. Then restore it to `0` for the
complete test split.

```bash
bash scripts/sft-3b-llm-judge/run_table10_qwen.sh \
  Qwen/Qwen2.5-VL-3B-Instruct sportsqa-qwen25vl-3b-yang

bash scripts/sft-3b-llm-judge/run_table10_qwen.sh \
  Qwen/Qwen2.5-VL-7B-Instruct sportsqa-qwen25vl-7b-yang

bash scripts/sft-3b-llm-judge/run_table10_qwen.sh \
  Qwen/Qwen2.5-VL-3B-Instruct sportsqa-qwen25vl-3b-lora-yang \
  /data/sportsqa/output/checkpoints/sportsqa-qwen25vl-3b/checkpoint-1000
```

The runner uses `yang-0s`, which passes only the original question alongside the video.
`infer_sportsqa.py --prompt-style yang-cot` is available for a separately labelled CoT
ablation; it is not part of the Qwen rows in Sports-QA Table 10.

## Fixed reproduction parameters

The parameters below are intentionally declared near the top of
`scripts/sft-3b-llm-judge/run_table10_qwen.sh`. They make a run comparable with a
later rerun, but the Sports-QA and Yang papers do not publish these Qwen video decoding
values. They are implementation choices, not claims about the original Qwen setup.

| Parameter | Current value | Effect and rationale |
| --- | ---: | --- |
| `max_new_tokens` | `32` | Maximum generated Qwen tokens for one QA. Sports-QA answers are short, so this prevents verbose output without cutting ordinary zero-shot answers. It is not suitable for a CoT experiment; give that separately named experiment a larger generation budget. |
| `video_frames` | `8` | Passed to the Qwen processor as `nframes`: eight decoded video frames represent each QA video. It balances temporal coverage, GPU memory, and visual-token length. The inference script requires an even value. |
| `video_min_pixels` | `50176` (`224 x 224`) | Lower visual-resolution budget used by the Qwen video processor. It prevents very small frames from being reduced below the fixed evaluation profile. |
| `video_max_pixels` | `200704` (`448 x 448`) | Upper visual-resolution budget used by the Qwen video processor. It limits visual tokens and GPU memory while retaining more detail than the lower bound. |
| `judge_batch_size` | `100` | Number of independent QA judgments included in one DeepSeek request. It improves judge throughput; each returned QA is still parsed and stored independently. |
| `judge_max_in_flight_batches` | `4` | At most four 100-QA judge requests are in flight at once. It reduces idle network time without changing the QA grouping or output file name. Start at `2` if the endpoint responds with rate-limit or timeout errors. |
| `judge_max_tokens` | `8192` | Maximum generated tokens for the *whole DeepSeek batch response*, not for one QA. It leaves room for 100 JSON judgments and short reasons; `1024` is too tight for this batch format. |
| `judge_timeout_seconds` | `240` | Maximum time to wait for one DeepSeek response. The judge defaults to 60 seconds, which can be too short for a large batch and trigger an unnecessary single-item fallback. |

Changing a video value, judge model, prompt, or `judge_batch_size` changes the evaluation
protocol. Use a new `run_name` and update `judge_label` when changing
`judge_batch_size`, so artifacts from unlike protocols are never mixed.
`judge_max_in_flight_batches` changes only request scheduling: it is recorded in the
artifacts but can be adjusted while resuming the same evaluation.

## Outputs

Each run writes beneath `${output_root}/${run_name}/`:

- `predictions/test_yang-0s.jsonl`: Qwen predictions and prompt style.
- `metrics/test_yang-0s_exact.json`: strict canonical-answer Accuracy and Macro-F1.
- `judgments/test_yang-0s_deepseek-v3-batch100.jsonl`: one auditable semantic judgment per QA.
- `metrics/test_yang-0s_deepseek-v3-batch100.json`: semantic Overall Accuracy, four question-type
  accuracies, average semantic score, and coverage.

The judge sends 100 independent QA items in one DeepSeek request by default, with up to
four requests in flight. It requires exactly one JSON judgment for every requested
`qa_id`; if a whole batch still fails after its retries, it automatically retries the
items one by one inside the same worker. The JSONL remains one auditable record per QA
and records its `batch_id`, actual `batch_size`, `judge_mode`, requested batch size, and
the concurrent-request limit. Worker threads never write JSONL; the main thread writes
and flushes only completed batches. Retries honor numeric `Retry-After` values when
available; otherwise they use exponential backoff with a small random jitter.

The runner enables `--resume` for both inference and semantic judging. Re-running the
same command after an interruption preserves existing predictions, skips completed
matching judgments, and retries missing or failed judgments. If every prediction already
exists, inference exits before loading the Qwen model. Missing or failed judgments make
the runner fail rather than silently treating them as incorrect. Resumed judge batches
continue their prior `batch_id` sequence. Change `judge_batch_size` and the
`judge_label` together if using a different batch size, so different protocols do not
share an artifact name.

The runner still creates the exact-match diagnostic files, but keeps their console output
silent. During semantic scoring, the terminal shows only the `DeepSeek semantic judging`
progress bar. On complete success, the final console JSON contains only `count`, `correct`,
and `accuracy`; consult the semantic metric file for question-type accuracies and average
semantic score.
