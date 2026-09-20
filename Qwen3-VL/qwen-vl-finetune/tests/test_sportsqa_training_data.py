import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def load_data_processor_module():
    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizer = object
    previous_transformers = sys.modules.get("transformers")
    module_name = "qwenvl.data.data_processor_under_test"
    sys.modules["transformers"] = transformers
    module_path = Path(__file__).parents[1] / "qwenvl" / "data" / "data_processor.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        if previous_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous_transformers


def load_validator_module():
    module_path = Path(__file__).parents[1] / "tools" / "validate_sportsqa.py"
    spec = importlib.util.spec_from_file_location("validate_sportsqa_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


data_processor = load_data_processor_module()
validator = load_validator_module()


class LazySupervisedDatasetFailureTests(unittest.TestCase):
    def make_dataset(self):
        dataset = object.__new__(data_processor.LazySupervisedDataset)
        dataset.list_data_dict = [
            {"qa_id": "broken", "video": "missing.mp4", "data_path": "/videos"},
            {"qa_id": "next", "video": "next.mp4", "data_path": "/videos"},
        ]
        return dataset

    def test_failed_sample_never_substitutes_the_next_sample(self):
        dataset = self.make_dataset()
        attempted_qa_ids = []

        def always_fail(sources):
            attempted_qa_ids.append(sources[0]["qa_id"])
            raise OSError("corrupt video")

        dataset.item_fn = always_fail
        with mock.patch.object(data_processor.time, "sleep"):
            with self.assertRaisesRegex(
                data_processor.DataSampleError,
                r"dataset_index=0.*qa_id=broken.*video=missing\.mp4.*Last error: corrupt video",
            ) as raised:
                dataset[0]

        self.assertEqual(attempted_qa_ids, ["broken"] * data_processor.SAMPLE_LOAD_RETRIES)
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_transient_failure_retries_the_original_sample(self):
        dataset = self.make_dataset()
        attempted_qa_ids = []

        def fail_twice_then_succeed(sources):
            attempted_qa_ids.append(sources[0]["qa_id"])
            if len(attempted_qa_ids) < 3:
                raise OSError("temporary storage error")
            return {"qa_id": sources[0]["qa_id"]}

        dataset.item_fn = fail_twice_then_succeed
        with mock.patch.object(data_processor.time, "sleep"):
            sample = dataset[0]

        self.assertEqual(sample, {"qa_id": "broken"})
        self.assertEqual(attempted_qa_ids, ["broken", "broken", "broken"])


class SportsQAValidationTests(unittest.TestCase):
    def test_validation_quarantines_only_invalid_records_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "present.mp4").write_bytes(b"placeholder")
            records = [
                {
                    "qa_id": "valid",
                    "video": "present.mp4",
                    "conversations": [{"from": "human", "value": "<video>"}],
                },
                {
                    "qa_id": "missing",
                    "video": "missing.mp4",
                    "conversations": [{"from": "human", "value": "<video>"}],
                },
            ]

            valid, quarantined, failure_counts, checked_video_count = validator.validate_records(
                records,
                video_root,
                verify_video=False,
                ffprobe_bin="ffprobe",
                timeout=30.0,
            )

        self.assertEqual([record["qa_id"] for record in valid], ["valid"])
        self.assertEqual(quarantined[0]["qa_id"], "missing")
        self.assertEqual(quarantined[0]["error_kind"], "missing_video")
        self.assertEqual(failure_counts, {"missing_video": 1})
        self.assertEqual(checked_video_count, 2)


if __name__ == "__main__":
    unittest.main()
