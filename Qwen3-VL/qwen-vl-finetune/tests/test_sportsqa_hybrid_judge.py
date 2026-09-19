import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError


def load_module(name, relative_path):
    module_path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


judge = load_module(
    "judge_sportsqa_semantic_under_test",
    Path("tools") / "sportsqa-qwen-llm-judge" / "judge_sportsqa_semantic.py",
)
summary = load_module(
    "summarize_sportsqa_semantic_under_test",
    Path("tools") / "sportsqa-qwen-llm-judge" / "summarize_sportsqa_semantic.py",
)


def item(qa_id, ans_cls):
    return {
        "qa_id": qa_id,
        "video_id": f"video-{qa_id}",
        "type": "what",
        "sport": "gymnastics",
        "question": "What happens?",
        "answer": "yes",
        "ans_cls": ans_cls,
    }


def prediction(qa_id, text):
    return {"qa_id": qa_id, "prediction_raw": text}


def rule_detail(qa_id, ans_cls, prediction_cls, raw_prediction):
    return {
        "qa_id": qa_id,
        "ans_cls": ans_cls,
        "prediction_cls": prediction_cls,
        "correct": prediction_cls == ans_cls,
        "prediction_raw": raw_prediction,
    }


class HybridJudgeTests(unittest.TestCase):
    def test_judge_request_disables_thinking(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{}"}}],"usage":{}}'

        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with patch.object(judge, "urlopen", side_effect=fake_urlopen):
            judge.call_chat_completion(
                "https://example.test/v1/chat/completions",
                "test-key",
                "Qwen/Qwen3-8B",
                "{}",
                0.0,
                64,
                1,
            )

        self.assertFalse(captured["payload"]["enable_thinking"])

    def test_rate_limiter_spaces_requests_by_tpm(self):
        limiter = judge.GlobalRateLimiter(rpm=60000, tpm=60000, safety_factor=1.0)
        start = time.monotonic()
        limiter.acquire("x", 99)
        limiter.acquire("x", 99)
        self.assertGreaterEqual(time.monotonic() - start, 0.08)

    def test_rate_limiter_applies_shared_cooldown(self):
        limiter = judge.GlobalRateLimiter(rpm=60000, tpm=6000000, safety_factor=1.0)
        limiter.cool_down(0.05)
        start = time.monotonic()
        limiter.acquire("x", 1)
        self.assertGreaterEqual(time.monotonic() - start, 0.04)

    def test_rate_limiter_releases_unused_token_reservation(self):
        limiter = judge.GlobalRateLimiter(rpm=60000, tpm=60000, safety_factor=1.0)
        reservation = limiter.acquire("x", 999)
        limiter.reconcile(reservation, {"total_tokens": 10})
        start = time.monotonic()
        limiter.acquire("x", 1)
        self.assertLess(time.monotonic() - start, 0.1)

    def test_429_extends_the_shared_cooldown_before_retrying(self):
        class RecordingLimiter:
            def __init__(self):
                self.acquire_count = 0
                self.cooldowns = []

            def acquire(self, user_content, max_tokens):
                self.acquire_count += 1
                return 1

            def cool_down(self, delay_seconds):
                self.cooldowns.append(delay_seconds)

            def reconcile(self, reserved_tokens, usage):
                pass

        args = SimpleNamespace(
            endpoint="https://example.test/v1/chat/completions",
            model="test-model",
            temperature=0.0,
            max_tokens=64,
            timeout_seconds=1,
            max_attempts=2,
            retry_backoff_seconds=2.0,
            max_retry_backoff_seconds=60.0,
            batch_size=1,
            max_in_flight_batches=1,
            rate_limit_rpm=1000,
            rate_limit_tpm=50000,
            rate_limit_safety_factor=0.8,
            rule_details=None,
        )
        limiter = RecordingLimiter()
        rate_limit_error = HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"message":"rate limited"}'),
        )
        with patch.object(
            judge,
            "call_chat_completion",
            side_effect=[
                rate_limit_error,
                ('{"judgments":[{"qa_id":"1","match":true,"score":5}]}', {}),
            ],
        ), patch.object(judge, "retry_delay_seconds", return_value=7.0), patch.object(
            judge.time, "sleep"
        ):
            rows, error = judge.request_batch_with_retries(
                [(item(1, 3), prediction(1, "yes"))],
                args,
                "test-key",
                1,
                "batch",
                limiter,
            )

        self.assertEqual(error, "")
        self.assertEqual(limiter.acquire_count, 2)
        self.assertEqual(limiter.cooldowns, [7.0])
        self.assertTrue(rows[0]["semantic_match"])

    def test_compact_llm_response_does_not_require_reason(self):
        parsed = judge.parse_judgments(
            '{"judgments":[{"qa_id":"1","match":true,"score":5}]}', ["1"]
        )
        self.assertEqual(parsed, {"1": (True, 5.0)})

    def test_rule_rows_only_cover_exact_canonical_predictions(self):
        correct_item = item(1, 3)
        correct_prediction = prediction(1, "correct")
        correct_row = judge.rule_row(
            correct_item,
            correct_prediction,
            rule_detail(1, 3, 3, "correct"),
        )
        self.assertEqual(correct_row["decision_source"], "rule_exact")
        self.assertTrue(correct_row["semantic_match"])
        self.assertEqual(correct_row["semantic_score"], 5.0)
        self.assertNotIn("judge_reason", correct_row)

        incorrect_row = judge.rule_row(
            item(2, 3),
            prediction(2, "wrong"),
            rule_detail(2, 3, 4, "wrong"),
        )
        self.assertFalse(incorrect_row["semantic_match"])
        self.assertEqual(incorrect_row["semantic_score"], 0.0)

        self.assertIsNone(
            judge.rule_row(
                item(3, 3),
                prediction(3, "a long free-form answer"),
                rule_detail(3, 3, None, "a long free-form answer"),
            )
        )

    def test_all_rule_run_needs_no_api_key_and_writes_hybrid_rows(self):
        manifest = [item(1, 3), item(2, 3)]
        predictions = [prediction(1, "correct"), prediction(2, "wrong")]
        details = [
            rule_detail(1, 3, 3, "correct"),
            rule_detail(2, 3, 4, "wrong"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest_path = tmp / "manifest.json"
            predictions_path = tmp / "predictions.jsonl"
            details_path = tmp / "details.jsonl"
            output_path = tmp / "judgments.jsonl"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
            )
            details_path.write_text(
                "".join(json.dumps(row) + "\n" for row in details), encoding="utf-8"
            )

            previous_argv = sys.argv
            try:
                sys.argv = [
                    "judge_sportsqa_semantic.py",
                    "--manifest", str(manifest_path),
                    "--predictions", str(predictions_path),
                    "--rule-details", str(details_path),
                    "--output-file", str(output_path),
                ]
                judge.main()
            finally:
                sys.argv = previous_argv

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["decision_source"] for row in rows}, {"rule_exact"})
        self.assertNotIn("judge_reason", rows[0])

    def test_hybrid_run_sends_only_noncanonical_predictions_to_llm(self):
        manifest = [item(1, 3), item(2, 3)]
        predictions = [prediction(1, "correct"), prediction(2, "free-form")]
        details = [
            rule_detail(1, 3, 3, "correct"),
            rule_detail(2, 3, None, "free-form"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest_path = tmp / "manifest.json"
            predictions_path = tmp / "predictions.jsonl"
            details_path = tmp / "details.jsonl"
            output_path = tmp / "judgments.jsonl"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
            )
            details_path.write_text(
                "".join(json.dumps(row) + "\n" for row in details), encoding="utf-8"
            )

            previous_argv = sys.argv
            previous_api_key = os.environ.get("SILICONFLOW_API_KEY")
            try:
                os.environ["SILICONFLOW_API_KEY"] = "test-key"
                sys.argv = [
                    "judge_sportsqa_semantic.py",
                    "--manifest", str(manifest_path),
                    "--predictions", str(predictions_path),
                    "--rule-details", str(details_path),
                    "--output-file", str(output_path),
                ]
                with patch.object(
                    judge,
                    "call_chat_completion",
                    return_value=(
                        '{"judgments":[{"qa_id":"2","match":true,"score":4.5}]}',
                        {"completion_tokens": 10},
                    ),
                ) as call:
                    judge.main()
            finally:
                sys.argv = previous_argv
                if previous_api_key is None:
                    os.environ.pop("SILICONFLOW_API_KEY", None)
                else:
                    os.environ["SILICONFLOW_API_KEY"] = previous_api_key

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(call.call_count, 1)
        self.assertEqual({row["decision_source"] for row in rows}, {"rule_exact", "llm_fallback"})
        llm_row = next(row for row in rows if row["decision_source"] == "llm_fallback")
        self.assertEqual(llm_row["semantic_score"], 4.5)
        self.assertNotIn("judge_reason", llm_row)

    def test_summary_reports_rule_and_llm_coverage(self):
        manifest = [item(1, 3), item(2, 3)]
        rows = [
            {
                "qa_id": 1,
                "type": "what",
                "status": "ok",
                "semantic_match": True,
                "semantic_score": 5.0,
                "judge_model": judge.RULE_MODEL,
                "judge_prompt_version": judge.RULE_PROTOCOL_VERSION,
                "judge_requested_batch_size": None,
                "judge_max_in_flight_batches": None,
                "decision_source": "rule_exact",
            },
            {
                "qa_id": 2,
                "type": "what",
                "status": "ok",
                "semantic_match": False,
                "semantic_score": 0.0,
                "judge_model": "deepseek",
                "judge_prompt_version": judge.PROMPT_VERSION,
                "judge_requested_batch_size": 100,
                "judge_max_in_flight_batches": 4,
                "judge_rate_limit_rpm": 1000,
                "judge_rate_limit_tpm": 50000,
                "judge_rate_limit_safety_factor": 0.8,
                "judge_enable_thinking": False,
                "decision_source": "llm_fallback",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest_path = tmp / "manifest.json"
            judgments_path = tmp / "judgments.jsonl"
            output_path = tmp / "metric.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            judgments_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            previous_argv = sys.argv
            try:
                sys.argv = [
                    "summarize_sportsqa_semantic.py",
                    "--manifest", str(manifest_path),
                    "--judgments", str(judgments_path),
                    "--output-file", str(output_path),
                    "--require-complete",
                ]
                summary.main()
            finally:
                sys.argv = previous_argv

            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["protocol"]["decision_source_counts"],
            {"llm_fallback": 1, "rule_exact": 1},
        )
        self.assertEqual(report["protocol"]["judge_rate_limit_rpms"], [1000])
        self.assertEqual(report["protocol"]["judge_rate_limit_tpms"], [50000])
        self.assertEqual(report["protocol"]["judge_enable_thinking_values"], [False])


if __name__ == "__main__":
    unittest.main()
