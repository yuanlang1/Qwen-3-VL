import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def load_infer_module():
    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = lambda *args, **kwargs: None
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = object
    transformers_models = types.ModuleType("transformers.models")
    transformers_auto = types.ModuleType("transformers.models.auto")
    modeling_auto = types.ModuleType("transformers.models.auto.modeling_auto")
    modeling_auto.AutoModelForImageTextToText = object
    modules = {
        "qwen_vl_utils": qwen_vl_utils,
        "transformers": transformers,
        "transformers.models": transformers_models,
        "transformers.models.auto": transformers_auto,
        "transformers.models.auto.modeling_auto": modeling_auto,
    }
    previous_modules = {name: sys.modules.get(name) for name in modules}
    previous_pipeline = sys.modules.pop("sportsqa_dataloader", None)
    sys.modules.update(modules)
    module_path = Path(__file__).parents[1] / "tools" / "infer_sportsqa.py"
    tools_path = str(module_path.parent)
    sys.path.insert(0, tools_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "infer_sportsqa_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(tools_path)
        if previous_pipeline is None:
            sys.modules.pop("sportsqa_dataloader", None)
        else:
            sys.modules["sportsqa_dataloader"] = previous_pipeline
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous_module


infer = load_infer_module()


def record(qa_id):
    return {
        "qa_id": qa_id,
        "video_id": f"video-{qa_id}",
        "type": "Descriptive",
        "sport": "sport",
        "question": f"Question {qa_id}?",
        "answer": "answer",
        "ans_cls": 1,
    }


class SportsQAInferenceTests(unittest.TestCase):
    def test_batch_size_groups_decoded_samples_only_in_main_process(self):
        samples = [
            infer.DecodedSample(
                sequence_id=index,
                record=record(index),
                message=[],
                video=torch.ones((1, 1)),
                video_metadata=None,
                sampled_fps=2.0,
                decode_ms=0.0,
            )
            for index in range(5)
        ]

        batches = list(infer.iter_gpu_batches(samples, batch_size=2))

        self.assertEqual([len(batch) for batch, _ in batches], [2, 2, 1])
        self.assertEqual(
            [sample.sequence_id for batch, _ in batches for sample in batch],
            [0, 1, 2, 3, 4],
        )

    def test_torchcodec_backend_is_checked_without_silent_fallback(self):
        with mock.patch.object(
            infer.importlib, "import_module", side_effect=OSError("missing libavcodec"),
        ) as import_module:
            with self.assertRaisesRegex(
                RuntimeError, r"TorchCodec could not load.*missing libavcodec"
            ):
                infer.configure_video_backend("torchcodec")

        import_module.assert_called_once_with("torchcodec.decoders")

    def test_backend_selection_is_propagated_to_spawned_workers(self):
        previous = os.environ.get("FORCE_QWENVL_VIDEO_READER")
        try:
            with mock.patch.object(infer.importlib, "import_module"):
                selected = infer.configure_video_backend("decord")
            self.assertEqual(selected, "decord")
            self.assertEqual(os.environ["FORCE_QWENVL_VIDEO_READER"], "decord")
        finally:
            if previous is None:
                os.environ.pop("FORCE_QWENVL_VIDEO_READER", None)
            else:
                os.environ["FORCE_QWENVL_VIDEO_READER"] = previous

    def test_resume_filter_preserves_manifest_order_without_duplicates(self):
        records = [record("one"), record("two"), record("three")]

        pending = infer.select_pending_records(records, {"two"}, limit=None)

        self.assertEqual([item["qa_id"] for item in pending], ["one", "three"])
        self.assertEqual(
            [
                item["qa_id"]
                for item in infer.select_pending_records(records, {"two"}, 1)
            ],
            ["one"],
        )

    def test_read_completed_ids_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"qa_id": 1}) + "\n\n" + json.dumps({"qa_id": "two"}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(infer.read_completed_ids(path), {"1", "two"})

    def test_write_predictions_keeps_batch_order_and_schema(self):
        batch = infer.PreparedBatch(
            sequence_ids=[0, 1],
            records=[record("one"), record("two")],
            inputs={},
            input_token_counts=[3, 4],
            decode_ms=0.0,
            processor_ms=0.0,
        )
        args = SimpleNamespace(
            model_name_or_path="model", adapter_path=None, prompt_style="current",
        )
        output = io.StringIO()

        infer.write_predictions(output, batch, ["first", "second"], args)

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["qa_id"] for row in rows], ["one", "two"])
        self.assertEqual([row["prediction_raw"] for row in rows], ["first", "second"])
        self.assertEqual(rows[0]["model_name_or_path"], "model")

    def test_decode_predictions_trims_each_left_padded_prompt(self):
        class FakeProcessor:
            def batch_decode(self, token_ids, **kwargs):
                self.kwargs = kwargs
                return [tokens.tolist() for tokens in token_ids]

        processor = FakeProcessor()
        inputs = {"input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]])}
        generated = torch.tensor([[0, 1, 2, 6], [3, 4, 5, 7]])

        predictions = infer.decode_predictions(processor, inputs, generated)

        self.assertEqual(predictions, [[6], [7]])
        self.assertEqual(
            processor.kwargs,
            {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
        )

    def test_legacy_gpu_feature_cache_paths_are_absent(self):
        tools_dir = Path(__file__).parents[1] / "tools"
        source = (tools_dir / "infer_sportsqa.py").read_text(encoding="utf-8")
        source += (tools_dir / "sportsqa_dataloader.py").read_text(encoding="utf-8")

        for legacy_name in (
            "cache-video-features",
            "CachedVideoFeatures",
            "OrderedBatchBuffer",
            "CudaPrefetcher",
            "reuse_cached_video_features",
        ):
            self.assertNotIn(legacy_name, source)


if __name__ == "__main__":
    unittest.main()
