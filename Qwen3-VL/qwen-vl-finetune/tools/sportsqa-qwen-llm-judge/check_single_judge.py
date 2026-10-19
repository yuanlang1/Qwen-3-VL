"""Offline regression check: python -B tools/sportsqa-qwen-llm-judge/check_single_judge.py."""

import io
import json
import subprocess
import sys
import threading
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import judge_sportsqa_semantic as judge
import summarize_sportsqa_semantic as summary


def run_main(module, arguments):
    with patch.object(sys, "argv", [module.__file__, *arguments]):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module.main()


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = [
            {"qa_id": str(index), "video_id": "demo", "type": "count",
             "sport": "demo", "question": "How many players?", "answer": "two", "ans_cls": 0}
            for index in range(1, 5)
        ]
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "answers.json").write_text(json.dumps({"two": 0, "three": 1}), encoding="utf-8")
        (root / "predictions.jsonl").write_text(
            "".join(json.dumps({"qa_id": item["qa_id"], "prediction_raw": raw}) + "\n"
                    for item, raw in zip(manifest, ["two", "three", "There are two players.", "unclear"])),
            encoding="utf-8",
        )
        subprocess.run([
            sys.executable, "-B", str(Path(judge.__file__).parent.parent / "eval_sportsqa.py"),
            "--manifest", str(root / "manifest.json"), "--predictions", str(root / "predictions.jsonl"),
            "--answer-map", str(root / "answers.json"), "--output-file", str(root / "exact.json"),
            "--details-file", str(root / "details.jsonl"),
        ], check=True, capture_output=True, text=True)

        arguments = [
            "--manifest", str(root / "manifest.json"), "--predictions", str(root / "predictions.jsonl"),
            "--rule-details", str(root / "details.jsonl"), "--output-file", str(root / "judgments.jsonl"),
            "--max-attempts", "2", "--retry-backoff-seconds", "0", "--max-in-flight-batches", "2", "--resume",
        ]
        counts = Counter()
        lock = threading.Lock()
        concurrent_requests = threading.Barrier(2, timeout=5)
        fail_four = True

        def fake_completion(endpoint, api_key, model, user_content, temperature, max_tokens, timeout_seconds):
            assert user_content.count("qa_id: ") == 1
            qa_id = user_content.splitlines()[0].split(": ", 1)[1]
            assert qa_id in {"3", "4"}, "Rule decisions must not call the API"
            with lock:
                counts[qa_id] += 1
                attempt = counts[qa_id]
            if attempt == 1:
                concurrent_requests.wait()
            if qa_id == "4" and fail_four:
                raise ValueError("simulated response failure")
            value = {"qa_id": qa_id, "pred": "yes" if qa_id == "3" else "no",
                     "score": 4.5 if qa_id == "3" else 0}
            return json.dumps({"judgments": [value]}), {"total_tokens": 1}

        pool = judge.AccountPool([judge.JudgeAccount("offline", "unused")], 10**12, 10**12, 1)
        with patch.object(judge, "call_chat_completion", side_effect=fake_completion):
            with patch.object(judge, "account_pool_for_pending_judgments", return_value=pool):
                run_main(judge, arguments)
                assert counts == {"3": 1, "4": 2}, counts
                rows = judge.load_jsonl(root / "judgments.jsonl")
                indexed = {row["qa_id"]: row for row in rows}
                assert indexed["1"]["semantic_score"] == 5
                assert indexed["2"]["semantic_score"] == 0
                assert indexed["3"]["semantic_match"] is True
                assert indexed["4"]["status"] == "error"
                base_keys = {"qa_id", "type", "prediction_raw", "prediction_sha256", "judge_model",
                             "judge_prompt_version", "decision_source", "status"}
                for row in rows:
                    result_keys = {"error"} if row["status"] == "error" else {"semantic_match", "semantic_score"}
                    assert set(row) == base_keys | result_keys, row

                summary_args = ["--manifest", str(root / "manifest.json"),
                                "--judgments", str(root / "judgments.jsonl"),
                                "--output-file", str(root / "metrics.json")]
                try:
                    run_main(summary, [*summary_args, "--require-complete"])
                except SystemExit as error:
                    assert "1 failed" in str(error)
                else:
                    raise AssertionError("Incomplete judgments must fail complete reporting")
                run_main(summary, summary_args)
                report = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
                assert report["coverage"] == 0.75
                assert report["failed_judgment_count"] == 1

                fail_four = False
                run_main(judge, arguments)
                assert counts == {"3": 1, "4": 3}, counts
                run_main(judge, arguments)
                assert counts == {"3": 1, "4": 3}, "Successful judgments must be skipped on resume"
                run_main(summary, [*summary_args, "--require-complete"])

        report = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        assert report["overall"] == {"count": 4, "correct": 2, "accuracy": 0.5, "average_score": 2.375}
        assert report["coverage"] == 1
        assert report["by_question_type"]["count"] == report["overall"]
        assert set(report["protocol"]) == {"name", "judge_models", "judge_prompt_versions", "decision_source_counts"}
        assert report["protocol"]["decision_source_counts"] == {"rule_exact": 2, "llm_fallback": 2}

        with patch.object(sys, "argv", [judge.__file__, *arguments, "--batch-size", "100"]):
            with redirect_stderr(io.StringIO()):
                try:
                    judge.parse_args()
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Judge batch size must no longer be configurable")

        for values in [[], [{"qa_id": "wrong", "pred": "yes", "score": 5}],
                       [{"qa_id": "3", "pred": "yes", "score": 5}] * 2,
                       [{"qa_id": "3", "pred": "yes", "score": 6}]]:
            try:
                judge.parse_judgments(json.dumps({"judgments": values}), ["3"])
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid single-QA response accepted")

        row = judge.load_completed_rows(root / "judgments.jsonl")["3"]
        for changes in [{"prediction_sha256": "changed"}, {"decision_source": "llm"},
                        {"judge_model": "changed"}, {"judge_prompt_version": "yang-video-semantic-batch-v5-yes-no-score"}]:
            try:
                judge.validate_existing_row({**row, **changes}, row["prediction_sha256"], "llm_fallback",
                                            judge.argparse.Namespace(model=judge.DEFAULT_MODEL), "3")
            except ValueError:
                pass
            else:
                raise AssertionError("Incompatible resume judgment accepted")

        runner = Path(judge.__file__).parents[2] / "scripts/sft-3b-llm-judge/run_table10_qwen.sh"
        script = runner.read_text(encoding="utf-8")
        assert "judge_batch_size" not in script
        assert '--batch-size "${batch_size}"' in script.split("judge_args=(")[0]
        assert "--batch-size" not in script.split("judge_args=(")[1]
        assert "judge_label=hybrid-rule-exact-yes-no-score-single" in script
        assert judge.SYSTEM_PROMPT in Path(judge.__file__).with_name("README.md").read_text(encoding="utf-8")
    print("Single-QA judge checks passed (mock API; no network requests).")


if __name__ == "__main__":
    main()
