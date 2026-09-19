"""Run deterministic Qwen-VL inference on a prepared Sports-QA manifest."""

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["qa_id"]))
    return completed


def configure_video_processor(processor: Any, args: argparse.Namespace) -> None:
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    for attribute, value in (
        ("min_pixels", args.video_min_pixels),
        ("max_pixels", args.video_max_pixels),
    ):
        if value is not None and hasattr(video_processor, attribute):
            setattr(video_processor, attribute, value)
    if hasattr(video_processor, "size") and isinstance(video_processor.size, dict):
        if args.video_min_pixels is not None:
            video_processor.size["shortest_edge"] = args.video_min_pixels
        if args.video_max_pixels is not None:
            video_processor.size["longest_edge"] = args.video_max_pixels


def resolve_video_path(video_root: Path, record: Dict[str, Any]) -> Path:
    path = Path(str(record["video"]))
    return path if path.is_absolute() else video_root / path


def group_records_by_video(
    records: Iterable[Dict[str, Any]], video_root: Path
) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    groups: Dict[Path, List[Dict[str, Any]]] = {}
    for record in records:
        video_path = resolve_video_path(video_root, record)
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video for qa_id={record['qa_id']}: {video_path}")
        groups.setdefault(video_path, []).append(record)
    return list(groups.items())


def build_video_content(video_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "type": "video",
        "video": str(video_path),
        "nframes": args.video_frames,
        "min_pixels": args.video_min_pixels,
        "max_pixels": args.video_max_pixels,
    }


def build_messages(
    records: Iterable[Dict[str, Any]], video_content: Dict[str, Any], prompt_style: str
) -> List[List[Dict[str, Any]]]:
    conversations = []
    for record in records:
        question = str(record["question"]).strip()
        if prompt_style == "current":
            prompt = f"Question: {question}\nAnswer with only the final answer."
        elif prompt_style == "yang-0s":
            prompt = question
        else:
            prompt = f"Let's think step by step. {question}"
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        dict(video_content),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
    return conversations


def chunks(records: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(records), batch_size):
        yield records[index : index + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--video-min-pixels", type=int, default=200704)
    parser.add_argument("--video-max-pixels", type=int, default=802816)
    parser.add_argument(
        "--cache-video-features",
        action="store_true",
        help="Encode each grouped video once and reuse its visual features across question batches.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("current", "yang-0s", "yang-cot"),
        default="current",
        help="Prompt protocol. current preserves the existing Sports-QA instruction.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_video_args(args: argparse.Namespace) -> None:
    if args.video_frames < 2:
        raise ValueError("--video-frames must be at least 2")
    if args.video_frames % 2:
        raise ValueError("--video-frames must be even")
    if args.video_min_pixels < 1 or args.video_max_pixels < args.video_min_pixels:
        raise ValueError("Video pixel limits must satisfy 1 <= min_pixels <= max_pixels")


@dataclass
class CachedVideo:
    frames: Any
    video_kwargs: Dict[str, Any]
    metadata: Any = None


@dataclass
class CachedVideoFeatures:
    video_features: Tuple[torch.Tensor, ...]
    deepstack_video_features: Optional[List[torch.Tensor]] = None


def load_video_once(video_content: Dict[str, Any], model_type: str) -> CachedVideo:
    conversation = [[{"role": "user", "content": [dict(video_content)]}]]
    if model_type == "qwen3_vl":
        _, videos, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is None or len(videos) != 1:
            raise ValueError("Expected exactly one decoded video")
        frames, metadata = videos[0]
        return CachedVideo(frames=frames, video_kwargs=video_kwargs, metadata=metadata)

    _, videos, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
    if videos is None or len(videos) != 1:
        raise ValueError("Expected exactly one decoded video")
    return CachedVideo(frames=videos[0], video_kwargs=video_kwargs)


def repeat_video_kwargs(video_kwargs: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    repeated_kwargs = dict(video_kwargs)
    fps = repeated_kwargs.get("fps")
    if isinstance(fps, list):
        if len(fps) != 1:
            raise ValueError("Expected one sampled fps value for a cached video")
        repeated_kwargs["fps"] = fps * batch_size
    return repeated_kwargs


def video_feature_model(model: Any) -> Any:
    """Return the Qwen-VL module whose forward method calls get_video_features."""
    candidates = []
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        candidates.append(get_base_model())
    candidates.append(model)

    for candidate in candidates:
        inner_model = getattr(candidate, "model", None)
        if hasattr(inner_model, "get_video_features"):
            return inner_model
        if hasattr(candidate, "get_video_features"):
            return candidate
    raise TypeError("The loaded model does not expose get_video_features().")


def first_video_inputs(inputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_values = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(video_grid_thw, torch.Tensor):
        raise ValueError("Video feature caching requires pixel_values_videos and video_grid_thw.")
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[0] < 1:
        raise ValueError("Expected video_grid_thw with at least one video entry.")

    first_grid = video_grid_thw[:1]
    first_video_patch_count = int(first_grid.prod().item())
    if pixel_values.shape[0] < first_video_patch_count:
        raise ValueError("pixel_values_videos does not contain the first video's complete patch sequence.")
    return pixel_values[:first_video_patch_count], first_grid


def encode_video_features_once(model: Any, inputs: Any, model_type: str) -> CachedVideoFeatures:
    """Encode the first (and only distinct) video in a grouped question batch."""
    if model_type not in {"qwen2_5_vl", "qwen3_vl"}:
        raise ValueError(f"Video feature caching is unsupported for model_type={model_type!r}.")

    pixel_values, video_grid_thw = first_video_inputs(inputs)
    encoded_features = video_feature_model(model).get_video_features(pixel_values, video_grid_thw)

    if model_type == "qwen2_5_vl":
        if not isinstance(encoded_features, tuple) or len(encoded_features) != 1:
            raise ValueError("Expected one Qwen2.5-VL video feature tensor.")
        return CachedVideoFeatures(video_features=encoded_features)

    if not isinstance(encoded_features, tuple) or len(encoded_features) != 2:
        raise ValueError("Expected Qwen3-VL video and deepstack feature tensors.")
    video_features, deepstack_video_features = encoded_features
    if not isinstance(video_features, tuple) or len(video_features) != 1:
        raise ValueError("Expected one Qwen3-VL video feature tensor.")
    if not isinstance(deepstack_video_features, list):
        raise ValueError("Expected Qwen3-VL deepstack features as a list.")
    return CachedVideoFeatures(
        video_features=video_features,
        deepstack_video_features=deepstack_video_features,
    )


def repeat_feature_for_batch(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
    return feature.repeat((batch_size,) + (1,) * (feature.ndim - 1))


@contextmanager
def reuse_cached_video_features(
    model: Any,
    cached_features: CachedVideoFeatures,
    model_type: str,
) -> Iterator[None]:
    """Make Qwen's regular multimodal forward reuse one video's visual features."""
    target_model = video_feature_model(model)
    original_get_video_features = target_model.get_video_features

    def cached_get_video_features(
        _pixel_values_videos: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Any:
        if video_grid_thw is None or video_grid_thw.ndim != 2:
            raise ValueError("Cached video features require a batched video_grid_thw.")
        batch_size = int(video_grid_thw.shape[0])
        if batch_size < 1:
            raise ValueError("Cached video features require at least one batch item.")

        repeated_video_features = tuple(cached_features.video_features[0] for _ in range(batch_size))
        if model_type == "qwen2_5_vl":
            return repeated_video_features
        if model_type == "qwen3_vl" and cached_features.deepstack_video_features is not None:
            repeated_deepstack_features = [
                repeat_feature_for_batch(feature, batch_size)
                for feature in cached_features.deepstack_video_features
            ]
            return repeated_video_features, repeated_deepstack_features
        raise ValueError(f"Cached video features are invalid for model_type={model_type!r}.")

    target_model.get_video_features = cached_get_video_features
    try:
        yield
    finally:
        target_model.get_video_features = original_get_video_features


def drop_repeated_video_pixels(inputs: Any) -> None:
    """Keep Qwen's video-input branch active without retaining repeated raw video patches."""
    pixel_values = inputs.get("pixel_values_videos")
    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError("Video feature caching requires pixel_values_videos.")
    inputs["pixel_values_videos"] = pixel_values.new_empty((0,) + tuple(pixel_values.shape[1:]))


def prepare_inputs_from_cached_video(
    processor: Any,
    model_type: str,
    messages: List[List[Dict[str, Any]]],
    cached_video: CachedVideo,
) -> Any:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch_size = len(messages)
    videos = [cached_video.frames] * batch_size
    video_kwargs = repeat_video_kwargs(cached_video.video_kwargs, batch_size)
    if model_type == "qwen3_vl":
        return processor(
            text=text,
            images=None,
            videos=videos,
            video_metadata=[cached_video.metadata] * batch_size,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

    return processor(
        text=text,
        images=None,
        videos=videos,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    validate_video_args(args)

    records = load_json(args.manifest)
    if not isinstance(records, list):
        raise ValueError("--manifest must contain a JSON list")
    completed_ids = read_completed_ids(args.output_file) if args.resume else set()
    records = [record for record in records if str(record["qa_id"]) not in completed_ids]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("Sports-QA inference: no pending QAs; existing predictions are unchanged.")
        return
    video_groups = group_records_by_video(records, args.video_root)

    # Intermediate LoRA checkpoints do not necessarily contain a processor.
    # The processor must match the immutable base model in every comparison.
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    configure_video_processor(processor, args)
    model_kwargs: Dict[str, Any] = {"dtype": args.dtype, "device_map": args.device_map}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    print(
        f"Video sampling: model_type={model.config.model_type}, frames={args.video_frames}; "
        f"prompt_style={args.prompt_style}; {len(records)} QAs across {len(video_groups)} unique videos"
    )
    if args.cache_video_features:
        print("Video feature cache: enabled (one visual encode per grouped video).")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output_file.open(mode, encoding="utf-8") as output_handle:
        with tqdm(total=len(records), desc="Sports-QA inference", unit="qa", dynamic_ncols=True) as progress:
            for video_index, (video_path, video_records) in enumerate(video_groups, start=1):
                video_content = build_video_content(video_path, args)
                cached_video = load_video_once(video_content, model.config.model_type)
                cached_video_features: Optional[CachedVideoFeatures] = None
                for batch_records in chunks(video_records, args.batch_size):
                    messages = build_messages(batch_records, video_content, args.prompt_style)
                    inputs = prepare_inputs_from_cached_video(
                        processor, model.config.model_type, messages, cached_video
                    )
                    input_token_count = inputs.input_ids.shape[-1]
                    max_context = getattr(model.config, "max_position_embeddings", None)
                    if max_context is None:
                        max_context = getattr(
                            getattr(model.config, "text_config", None), "max_position_embeddings", None
                        )
                    if max_context is not None and input_token_count + args.max_new_tokens > max_context:
                        raise RuntimeError(
                            "Video preprocessing produced "
                            f"{input_token_count} input tokens, which leaves insufficient context for "
                            f"{args.max_new_tokens} generated tokens (model limit: {max_context}). "
                            "Reduce --video-frames or --video-max-pixels."
                        )
                    inputs = inputs.to(model.device)
                    generation_kwargs: Dict[str, Any] = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.temperature > 0,
                    }
                    if args.temperature > 0:
                        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
                    with torch.inference_mode():
                        if args.cache_video_features:
                            if cached_video_features is None:
                                cached_video_features = encode_video_features_once(
                                    model, inputs, model.config.model_type
                                )
                            drop_repeated_video_pixels(inputs)
                            with reuse_cached_video_features(
                                model, cached_video_features, model.config.model_type
                            ):
                                generated_ids = model.generate(**inputs, **generation_kwargs)
                        else:
                            generated_ids = model.generate(**inputs, **generation_kwargs)
                    trimmed_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    predictions = processor.batch_decode(
                        trimmed_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for record, prediction in zip(batch_records, predictions):
                        result = {
                            "qa_id": record["qa_id"],
                            "video_id": record["video_id"],
                            "type": record["type"],
                            "sport": record["sport"],
                            "question": record["question"],
                            "answer": record["answer"],
                            "ans_cls": record["ans_cls"],
                            "prediction_raw": prediction,
                            "model_name_or_path": args.model_name_or_path,
                            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                            "prompt_style": args.prompt_style,
                        }
                        output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    progress.update(len(batch_records))
                    progress.set_postfix(
                        tokens=input_token_count,
                        videos=f"{video_index}/{len(video_groups)}",
                    )
                del cached_video
                del cached_video_features


if __name__ == "__main__":
    main()
