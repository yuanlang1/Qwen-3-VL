import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def load_infer_module():
    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = lambda *args, **kwargs: None
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = object
    transformers.models = types.ModuleType("transformers.models")
    transformers.models.auto = types.ModuleType("transformers.models.auto")
    modeling_auto = types.ModuleType("transformers.models.auto.modeling_auto")
    modeling_auto.AutoModelForImageTextToText = object

    modules = {
        "qwen_vl_utils": qwen_vl_utils,
        "transformers": transformers,
        "transformers.models": transformers.models,
        "transformers.models.auto": transformers.models.auto,
        "transformers.models.auto.modeling_auto": modeling_auto,
    }
    previous_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = Path(__file__).parents[1] / "tools" / "infer_sportsqa.py"
        spec = importlib.util.spec_from_file_location("infer_sportsqa_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous_module


infer = load_infer_module()


class FakeQwen2Model:
    def __init__(self):
        self.calls = 0

    def get_video_features(self, pixels, grid):
        self.calls += 1
        return (pixels.clone(),)


class FakeQwen3Model:
    def __init__(self):
        self.calls = 0

    def get_video_features(self, pixels, grid):
        self.calls += 1
        return (pixels.clone(),), [pixels[:, :1].clone(), pixels[:, :1].clone()]


class FakeGenerationWrapper:
    def __init__(self, inner_model):
        self.model = inner_model

    def get_video_features(self, pixels, grid):
        return self.model.get_video_features(pixels, grid)


def batch_inputs(batch_size=2):
    patches_per_video = 8
    return {
        "pixel_values_videos": torch.arange(
            batch_size * patches_per_video, dtype=torch.float32
        ).reshape(-1, 1),
        "video_grid_thw": torch.tensor([[2, 2, 2]] * batch_size),
    }


class VideoFeatureCacheTests(unittest.TestCase):
    def test_qwen2_features_are_encoded_once_and_reused_per_batch_item(self):
        model = FakeQwen2Model()
        inputs = batch_inputs(batch_size=2)

        cached = infer.encode_video_features_once(model, inputs, "qwen2_5_vl")
        self.assertEqual(model.calls, 1)
        infer.drop_repeated_video_pixels(inputs)
        self.assertEqual(inputs["pixel_values_videos"].shape[0], 0)

        with infer.reuse_cached_video_features(model, cached, "qwen2_5_vl"):
            repeated = model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        self.assertEqual(len(repeated), 2)
        self.assertTrue(torch.equal(repeated[0], cached.video_features[0]))
        self.assertTrue(torch.equal(repeated[1], cached.video_features[0]))
        self.assertEqual(model.calls, 1)

    def test_qwen3_deepstack_features_are_repeated_with_video_features(self):
        model = FakeQwen3Model()
        inputs = batch_inputs(batch_size=3)

        cached = infer.encode_video_features_once(model, inputs, "qwen3_vl")
        self.assertEqual(model.calls, 1)
        with infer.reuse_cached_video_features(model, cached, "qwen3_vl"):
            repeated_features, repeated_deepstack = model.get_video_features(
                inputs["pixel_values_videos"], inputs["video_grid_thw"]
            )
        self.assertEqual(len(repeated_features), 3)
        self.assertEqual(len(repeated_deepstack), 2)
        self.assertEqual(repeated_deepstack[0].shape[0], 3 * cached.deepstack_video_features[0].shape[0])
        self.assertTrue(
            torch.equal(
                repeated_deepstack[0][: cached.deepstack_video_features[0].shape[0]],
                cached.deepstack_video_features[0],
            )
        )
        self.assertEqual(model.calls, 1)

    def test_inner_qwen_module_is_patched_not_the_generation_wrapper(self):
        inner_model = FakeQwen2Model()
        model = FakeGenerationWrapper(inner_model)
        inputs = batch_inputs(batch_size=2)

        self.assertIs(infer.video_feature_model(model), inner_model)
        cached = infer.encode_video_features_once(model, inputs, "qwen2_5_vl")
        self.assertEqual(inner_model.calls, 1)
        with infer.reuse_cached_video_features(model, cached, "qwen2_5_vl"):
            repeated = inner_model.get_video_features(
                inputs["pixel_values_videos"], inputs["video_grid_thw"]
            )
        self.assertEqual(len(repeated), 2)
        self.assertEqual(inner_model.calls, 1)


if __name__ == "__main__":
    unittest.main()
