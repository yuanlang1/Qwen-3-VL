"""CPU regression checks; no model weights, video decoder, or API calls needed."""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# This module only uses Transformers for a tokenizer type annotation. Exercise
# its real preprocessing and dataset code without installing the training stack.
with patch.dict(sys.modules, {"transformers": SimpleNamespace(PreTrainedTokenizer=object)}):
    data = importlib.import_module("qwenvl.data.data_processor")


class Processor:
    def __init__(self, max_length=8192, answer=True):
        self.tokenizer = SimpleNamespace(
            model_max_length=max_length, pad_token_id=0, decode=lambda *a, **k: ""
        )
        self.video_processor = SimpleNamespace(temporal_patch_size=2, fps=2)
        self.answer = answer
        self.calls = []

    def result(self, seconds=1):
        ids = [151652, 151656, 151653]
        if self.answer:
            ids += [77091, 198, 42, 151645, 198]
        return {
            "input_ids": torch.tensor([ids]),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "second_per_grid_ts": [seconds],
        }

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.result() if kwargs["tokenize"] else "training conversation"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return self.result()


class FixedVideoSFTTests(unittest.TestCase):
    def setUp(self):
        self.source = [{
            "video": "clip.mp4", "data_path": "/videos",
            "conversations": [
                {"from": "human", "value": "<video>\nQuestion?"},
                {"from": "gpt", "value": "four"},
            ],
        }]

    def args(self, frames=8, pixels=200704):
        return SimpleNamespace(
            video_nframes=frames, video_min_pixels=50176, video_max_pixels=pixels
        )

    def decode(self, frames):
        def process(messages):
            self.decoded_messages = messages
            return None, [torch.zeros(frames, 3, 28, 28)]
        return patch.dict(sys.modules, {
            "qwen_vl_utils": SimpleNamespace(process_vision_info=process)
        })

    def test_all_four_profiles_preserve_budget_and_answer_labels(self):
        for frames in (8, 16):
            for pixels in (200704, 401408):
                with self.subTest(frames=frames, pixels=pixels), self.decode(frames):
                    processor = Processor()
                    result = data.preprocess_qwen_visual(
                        self.source, processor, self.args(frames, pixels)
                    )
                    content = self.decoded_messages[0]["content"][0]
                    self.assertEqual(content["nframes"], frames)
                    self.assertEqual(content["min_pixels"], 50176)
                    self.assertEqual(content["max_pixels"], pixels)
                    self.assertNotIn("fps", content)
                    self.assertFalse(processor.inputs["do_resize"])
                    self.assertFalse(processor.inputs["do_sample_frames"])
                    self.assertNotIn("fps", processor.inputs)
                    self.assertEqual(result["labels"][0, :5].tolist(), [-100] * 5)
                    self.assertEqual(result["labels"][0, 5:].tolist(), [42, 151645, 198])

    def test_temporal_positions_use_processor_default(self):
        dataset = object.__new__(data.LazySupervisedDataset)
        dataset.processor = Processor()
        dataset.data_args = self.args()
        dataset.merge_size = 2
        captured = {}

        def rope(*args, **kwargs):
            captured.update(kwargs)
            return torch.zeros(3, 1, 8, dtype=torch.long), None

        dataset.get_rope_index = rope
        with self.decode(8):
            result = dataset._get_item(self.source)
        self.assertEqual(captured["second_per_grid_ts"], [1])
        self.assertNotIn("second_per_grid_ts", result)

    def test_wrong_frame_count_fails(self):
        with self.decode(6), self.assertRaisesRegex(ValueError, "exactly 8"):
            data.preprocess_qwen_visual(self.source, Processor(), self.args())

    def test_overlong_sample_fails_before_truncation(self):
        with self.decode(8), self.assertRaisesRegex(ValueError, "exceeding"):
            data.preprocess_qwen_visual(self.source, Processor(max_length=4), self.args())

    def test_missing_supervision_fails(self):
        with self.decode(8), self.assertRaisesRegex(ValueError, "no supervised"):
            data.preprocess_qwen_visual(self.source, Processor(answer=False), self.args())

    def test_legacy_fps_path_is_preserved(self):
        processor = Processor()
        result = data.preprocess_qwen_visual(self.source, processor)
        self.assertTrue(processor.calls[0][1]["tokenize"])
        self.assertNotIn("nframes", processor.calls[0][0][0]["content"][0])
        self.assertTrue((result["labels"] != -100).any())


if __name__ == "__main__":
    unittest.main()
