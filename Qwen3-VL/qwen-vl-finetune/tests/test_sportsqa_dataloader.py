import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def load_pipeline_module():
    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = lambda *args, **kwargs: None
    modules = {"qwen_vl_utils": qwen_vl_utils}
    previous_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = Path(__file__).parents[1] / "tools" / "sportsqa_dataloader.py"
        spec = importlib.util.spec_from_file_location(
            "sportsqa_dataloader_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous_module


pipeline = load_pipeline_module()


class FakeProcessor:
    def __init__(self, token_counts=None):
        self.token_counts = token_counts
        self.calls = []

    def apply_chat_template(self, messages, **_):
        return [f"prompt-{index}" for index in range(len(messages))]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = len(kwargs["text"])
        counts = self.token_counts or [3] * batch_size
        max_length = max(counts)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        for index, count in enumerate(counts):
            attention_mask[index, -count:] = 1
        return {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
            "attention_mask": attention_mask,
            "pixel_values_videos": torch.ones((batch_size, 2), dtype=torch.float32),
            "video_grid_thw": torch.ones((batch_size, 3), dtype=torch.long),
        }


def record(qa_id, video="sport/video.mp4"):
    return {
        "qa_id": qa_id,
        "video_id": "sport/video",
        "video": video,
        "type": "Descriptive",
        "sport": "sport",
        "question": f"Question {qa_id}?",
        "answer": "answer",
        "ans_cls": 1,
    }


def pipeline_config(model_type="qwen2_5_vl", max_context=64, max_new_tokens=32):
    return pipeline.PipelineConfig(
        model_type=model_type,
        prompt_style="current",
        video_frames=8,
        video_min_pixels=50176,
        video_max_pixels=200704,
        max_new_tokens=max_new_tokens,
        max_context=max_context,
    )


def work(sequence_id, qa_id, video_path="video.mp4"):
    return pipeline.QAWork(
        sequence_id=sequence_id,
        video_path=Path(video_path),
        record=record(qa_id, video_path),
    )


class SportsQADataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.original_process_vision_info = pipeline.process_vision_info
        self.processor = FakeProcessor()

    def tearDown(self):
        pipeline.process_vision_info = self.original_process_vision_info

    def test_same_video_is_decoded_for_each_qa_in_manifest_order(self):
        calls = []

        def fake_process(_message, **kwargs):
            calls.append(kwargs)
            value = float(len(calls))
            return None, [torch.full((2, 2), value)], {"fps": [value]}

        pipeline.process_vision_info = fake_process
        config = pipeline_config()
        decoder = pipeline.SportsQADecoder(config)
        samples = [decoder(work(0, 1)), decoder(work(1, 2))]
        prepared = pipeline.prepare_batch(samples, self.processor, config)

        self.assertEqual(len(calls), 2)
        self.assertEqual(prepared.sequence_ids, [0, 1])
        self.assertEqual([item["qa_id"] for item in prepared.records], [1, 2])
        processor_kwargs = self.processor.calls[0]
        self.assertEqual(processor_kwargs["fps"], [1.0, 2.0])
        self.assertEqual(
            [float(video[0, 0]) for video in processor_kwargs["videos"]], [1.0, 2.0]
        )
        self.assertFalse(processor_kwargs["do_resize"])
        self.assertFalse(processor_kwargs["do_sample_frames"])

    def test_qwen3_passes_video_metadata_without_fps(self):
        process_kwargs = []

        def fake_process(_message, **kwargs):
            process_kwargs.append(kwargs)
            index = len(process_kwargs)
            return (
                None,
                [(torch.ones((2, 2)), {"clip": index})],
                {"do_sample_frames": False},
            )

        pipeline.process_vision_info = fake_process
        config = pipeline_config(model_type="qwen3_vl")
        decoder = pipeline.SportsQADecoder(config)
        samples = [decoder(work(0, 1)), decoder(work(1, 2))]
        pipeline.prepare_batch(samples, self.processor, config)

        self.assertEqual(
            process_kwargs,
            [
                {
                    "return_video_kwargs": True,
                    "image_patch_size": 16,
                    "return_video_metadata": True,
                },
                {
                    "return_video_kwargs": True,
                    "image_patch_size": 16,
                    "return_video_metadata": True,
                },
            ],
        )
        processor_kwargs = self.processor.calls[0]
        self.assertEqual(processor_kwargs["video_metadata"], [{"clip": 1}, {"clip": 2}])
        self.assertNotIn("fps", processor_kwargs)

    def test_decode_error_keeps_exact_qa_context(self):
        def broken_process(*_args, **_kwargs):
            raise OSError("corrupt video")

        pipeline.process_vision_info = broken_process
        with self.assertRaisesRegex(
            pipeline.VideoPreparationError,
            r"video decode.*sequence_id=7.*qa_id=11.*broken\.mp4.*corrupt video",
        ):
            pipeline.SportsQADecoder(pipeline_config())(work(7, 11, "broken.mp4"))

    def test_processor_error_lists_every_qa_in_the_batch(self):
        class BrokenProcessor(FakeProcessor):
            def __call__(self, **_kwargs):
                raise ValueError("bad processor input")

        self.processor = BrokenProcessor()
        pipeline.process_vision_info = lambda *_args, **_kwargs: (
            None,
            [torch.ones((2, 2))],
            {"fps": [2.0]},
        )
        config = pipeline_config()
        decoder = pipeline.SportsQADecoder(config)
        samples = [decoder(work(0, 1)), decoder(work(1, 2))]
        with self.assertRaisesRegex(
            pipeline.VideoPreparationError,
            r"processor preparation.*sequence_id=0, qa_id=1.*sequence_id=1, qa_id=2",
        ):
            pipeline.prepare_batch(samples, self.processor, config)

    def test_context_validation_uses_each_unpadded_token_count(self):
        self.processor = FakeProcessor(token_counts=[3, 5])
        pipeline.process_vision_info = lambda *_args, **_kwargs: (
            None,
            [torch.ones((2, 2))],
            {"fps": [2.0]},
        )
        config = pipeline_config(max_context=6, max_new_tokens=2)
        decoder = pipeline.SportsQADecoder(config)
        samples = [decoder(work(0, 1)), decoder(work(1, 2))]
        with self.assertRaisesRegex(
            pipeline.VideoPreparationError,
            r"context validation.*sequence_id=1.*qa_id=2.*5 input tokens",
        ):
            pipeline.prepare_batch(samples, self.processor, config)

    def test_dataloader_prefetches_single_samples_not_gpu_batches(self):
        items = [work(0, 1), work(1, 2), work(2, 3)]
        dataset = pipeline.SportsQADataset(items)
        args = SimpleNamespace(
            batch_size=2,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=True,
            timeout=120,
            multiprocessing_context="spawn",
            decoder_threads=2,
        )
        loader = pipeline.make_dataloader(
            dataset, pipeline.SportsQADecoder(pipeline_config()), args,
        )

        self.assertEqual(
            [dataset[index].sequence_id for index in range(len(dataset))], [0, 1, 2]
        )
        self.assertIsNone(loader.batch_size)
        self.assertIsNone(loader.batch_sampler)
        self.assertEqual(loader.prefetch_factor, 2)
        self.assertFalse(loader.persistent_workers)

    def test_work_plan_validates_paths_without_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            items = pipeline.build_qa_work_items(
                [record(1, "video.mp4"), record(2, "video.mp4")], root,
            )

            self.assertEqual([item.sequence_id for item in items], [0, 1])
            self.assertEqual([item.video_path for item in items], [video, video])
            with self.assertRaisesRegex(FileNotFoundError, r"sequence_id=0.*qa_id=3"):
                pipeline.build_qa_work_items([record(3, "missing.mp4")], root)

    def test_tensor_device_move_preserves_nested_inputs(self):
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "nested": {
                "video": torch.ones((1, 2)),
                "grid": [torch.ones((1, 3), dtype=torch.long)],
            },
        }

        moved = pipeline.move_inputs_to_device(inputs, "cpu")

        self.assertEqual(moved["input_ids"].device.type, "cpu")
        self.assertEqual(moved["nested"]["video"].device.type, "cpu")
        self.assertEqual(moved["nested"]["grid"][0].device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Pinned-memory test requires CUDA")
    def test_prepared_batch_pins_all_model_tensors(self):
        prepared = pipeline.PreparedBatch(
            sequence_ids=[0],
            records=[record(1)],
            inputs={
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "pixel_values_videos": torch.ones((1, 2)),
                "video_grid_thw": torch.ones((1, 3), dtype=torch.long),
            },
            input_token_counts=[2],
            decode_ms=0.0,
            processor_ms=0.0,
        )

        prepared.pin_memory()

        self.assertTrue(prepared.inputs["input_ids"].is_pinned())
        self.assertTrue(prepared.inputs["pixel_values_videos"].is_pinned())
        self.assertTrue(prepared.inputs["video_grid_thw"].is_pinned())


if __name__ == "__main__":
    unittest.main()
