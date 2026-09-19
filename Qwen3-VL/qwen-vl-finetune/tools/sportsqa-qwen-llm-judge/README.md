# Sports-QA Qwen LLM Judge

This directory contains the isolated Table 10-style Qwen evaluation workflow:

1. Qwen video inference on the Sports-QA `test` split with the Yang zero-shot prompt.
2. Existing deterministic 191-class exact-match scoring for diagnostics.
3. Hybrid semantic scoring: exact answer-vocabulary rules first, then SiliconFlow
   `deepseek-ai/DeepSeek-V3` only for free-form predictions that rules cannot classify.

The Hybrid score is **not** directly comparable to the previous all-LLM semantic score:
it uses deterministic exact-match decisions before LLM fallback. It is also not an exact
reproduction of Yang et al., which used GPT-4 as the judge. Keep the metric files separate.

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

## Continue judging completed inference

The normal runner starts exact scoring and LLM judging immediately after inference exits
successfully. If inference completed in an earlier invocation, continue only the judging
stages without loading Qwen or requiring a model/adapter argument:

```bash
bash scripts/sft-3b-llm-judge/run_table10_qwen.sh \
  --judge-only sportsqa-qwen25vl-3b-yang
```

This expects the existing prediction file at
`${output_root}/${run_name}/predictions/test_yang-0s.jsonl`, regenerates the deterministic
exact-match details, then resumes any missing LLM judgments. It fails explicitly if the
prediction file is absent; the judge itself still checks that it covers the requested
manifest before making API calls.

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
| `cache_video_features` | `false` | Disabled for formal evaluation because the experimental cross-batch visual-feature cache changed generated answers in the cache/no-cache equivalence check. |
| `judge_batch_size` | `100` | Number of independent fallback QA judgments included in one DeepSeek request. Exact canonical predictions bypass DeepSeek. |
| `judge_max_in_flight_batches` | `4` | At most four 100-QA judge requests are in flight at once. It reduces idle network time without changing the QA grouping or output file name. Start at `2` if the endpoint responds with rate-limit or timeout errors. |
| `judge_rate_limit_rpm` | `1000` | SiliconFlow L0 account RPM ceiling. Every DeepSeek request, including retries and single-item fallbacks, shares this process-wide limit. |
| `judge_rate_limit_tpm` | `50000` | SiliconFlow L0 account TPM ceiling. Each request conservatively reserves its UTF-8 input size plus `judge_max_tokens`. |
| `judge_rate_limit_safety_factor` | `0.8` | Uses 80% of the L0 ceilings (`800` RPM and `40,000` TPM) to leave account-level headroom. |
| `judge_enable_thinking` | `false` | The judge explicitly disables Qwen3 thinking mode. This avoids hidden reasoning-token latency for a constrained JSON classification task. |
| `judge_max_tokens` | `4096` | Maximum generated tokens for the *whole DeepSeek batch response*. The compact response contains only `qa_id`, `match`, and `score`; no reason is requested or stored. |
| `judge_timeout_seconds` | `240` | Maximum time to wait for one DeepSeek response. The judge defaults to 60 seconds, which can be too short for a large batch and trigger an unnecessary single-item fallback. |

Changing a video value, judge model, prompt, rule policy, or `judge_batch_size` changes
the evaluation protocol. Use a new `run_name` and update `judge_label` when changing
them, so artifacts from unlike protocols are never mixed.
`judge_max_in_flight_batches` changes only request scheduling: it is recorded in the
artifacts but can be adjusted while resuming the same evaluation.

## Outputs

Each run writes beneath `${output_root}/${run_name}/`:

- `predictions/test_yang-0s.jsonl`: Qwen predictions and prompt style.
- `metrics/test_yang-0s_exact.json`: strict canonical-answer Accuracy and Macro-F1.
- `judgments/test_yang-0s_hybrid-rule-exact-deepseek-v3-compact-no-thinking-batch100.jsonl`: one hybrid judgment per QA.
- `metrics/test_yang-0s_hybrid-rule-exact-deepseek-v3-compact-no-thinking-batch100.json`: hybrid Overall Accuracy,
  question-type accuracies, average score, coverage, and rule/LLM decision-source counts.

The hybrid judge first reuses the exact-match detail file. When a prediction exactly
normalizes to one answer-vocabulary label, it is recorded as `decision_source=rule_exact`;
all other predictions are sent to DeepSeek as the fallback queue. The compact DeepSeek
response requires only `qa_id`, `match`, and `score`. It sends up to 100 fallback QA items
per request and four requests in flight. A failed batch retries its items one by one. The
JSONL records `decision_source`, `batch_id`, actual `batch_size`, `judge_mode`, requested
batch size, the concurrent-request limit, and the configured rate ceilings. Worker
threads never write JSONL; the main thread writes and flushes only completed batches.
Retries honor numeric `Retry-After` values when available; otherwise they use exponential
backoff with a small random jitter.

The online judge also uses a process-wide leaky-bucket limiter. Before every DeepSeek
request, all workers share both the RPM and TPM schedule; the next request starts only
after both budgets permit it. A `429` extends one shared cooldown, so workers that did
not receive the error pause too. The limiter uses the configured ceilings with the safety
factor and initially reserves UTF-8 input bytes plus `max_tokens`. After a successful
response, it refunds the unused reservation using `usage.total_tokens`, allowing later
requests to follow observed consumption rather than the generation cap. This prevents
bursty local traffic but cannot coordinate separate processes or other machines using the
same account; run one online evaluator per account or lower the configured ceilings to
reserve capacity for those callers.

The runner enables `--resume` for both inference and hybrid judging. Re-running the
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
