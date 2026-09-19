"""CPU DataLoader pipeline for deterministic Sports-QA Qwen-VL inference."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PipelineConfig:
    """Pickle-safe video and model-input settings."""

    model_type: str
    prompt_style: str
    video_frames: int
    video_min_pixels: int
    video_max_pixels: int
    max_new_tokens: int
    max_context: Optional[int]


@dataclass(frozen=True)
class QAWork:
    """One independent QA item in manifest order."""

    sequence_id: int
    video_path: Path
    record: Dict[str, Any]


@dataclass
class DecodedSample:
    """One QA after CPU video decode, before model input preparation."""

    sequence_id: int
    record: Dict[str, Any]
    message: List[Dict[str, Any]]
    video: Any
    video_metadata: Any
    sampled_fps: Optional[float]
    decode_ms: float


@dataclass
class PreparedBatch:
    """One GPU batch after processor preparation in the main process."""

    sequence_ids: List[int]
    records: List[Dict[str, Any]]
    inputs: Dict[str, Any]
    input_token_counts: List[int]
    decode_ms: float
    processor_ms: float

    def pin_memory(self) -> "PreparedBatch":
        self.inputs = transform_tensors(self.inputs, lambda value: value.pin_memory())
        return self


class VideoPreparationError(RuntimeError):
    """Preserve enough context to diagnose a failing CPU or processor stage."""


def transform_tensors(value: Any, transform: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, Mapping):
        return {key: transform_tensors(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(transform_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [transform_tensors(item, transform) for item in value]
    return value


def move_inputs_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return transform_tensors(inputs, lambda value: value.to(device, non_blocking=True))


def configure_video_processor(processor: Any, config: PipelineConfig) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    for attribute, value in (
        ("min_pixels", config.video_min_pixels),
        ("max_pixels", config.video_max_pixels),
    ):
        if hasattr(video_processor, attribute):
            setattr(video_processor, attribute, value)
    if hasattr(video_processor, "size") and isinstance(video_processor.size, dict):
        video_processor.size["shortest_edge"] = config.video_min_pixels
        video_processor.size["longest_edge"] = config.video_max_pixels


def resolve_video_path(video_root: Path, record: Dict[str, Any]) -> Path:
    path = Path(str(record["video"]))
    return path if path.is_absolute() else video_root / path


def build_qa_work_items(
    records: Iterable[Dict[str, Any]], video_root: Path
) -> List[QAWork]:
    work_items = []
    for sequence_id, record in enumerate(records):
        video_path = resolve_video_path(video_root, record)
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Missing video for sequence_id={sequence_id}, "
                f"qa_id={record['qa_id']}: {video_path}"
            )
        work_items.append(
            QAWork(sequence_id=sequence_id, video_path=video_path, record=record)
        )
    return work_items


def build_video_content(work: QAWork, config: PipelineConfig) -> Dict[str, Any]:
    return {
        "type": "video",
        "video": str(work.video_path),
        "nframes": config.video_frames,
        "min_pixels": config.video_min_pixels,
        "max_pixels": config.video_max_pixels,
    }


def build_message(work: QAWork, config: PipelineConfig) -> List[Dict[str, Any]]:
    question = str(work.record["question"]).strip()
    if config.prompt_style == "current":
        prompt = f"Question: {question}\nAnswer with only the final answer."
    elif config.prompt_style == "yang-0s":
        prompt = question
    else:
        prompt = f"Let's think step by step. {question}"
    return [
        {
            "role": "user",
            "content": [
                build_video_content(work, config),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _work_error(work: QAWork, stage: str, error: Exception) -> VideoPreparationError:
    return VideoPreparationError(
        f"Sports-QA {stage} failed for sequence_id={work.sequence_id}, "
        f"qa_id={work.record['qa_id']}, video_path={work.video_path}: {error}"
    )


def _sample_context(sample: DecodedSample) -> str:
    return (
        f"sequence_id={sample.sequence_id}, qa_id={sample.record['qa_id']}, "
        f"video_path={sample.message[0]['content'][0]['video']}"
    )


class SportsQADataset(Dataset):
    """Map-style dataset whose items are independent QA records."""

    def __init__(self, work_items: Sequence[QAWork]) -> None:
        self._work_items = list(work_items)

    def __len__(self) -> int:
        return len(self._work_items)

    def __getitem__(self, index: int) -> QAWork:
        return self._work_items[index]


class SportsQADecoder:
    """Decode one QA video per DataLoader task."""

    def __init__(self, config: PipelineConfig) -> None:
        if config.model_type not in {"qwen2_5_vl", "qwen3_vl"}:
            raise ValueError(f"Unsupported Sports-QA model_type={config.model_type!r}")
        self._config = config

    def __call__(self, work: QAWork) -> DecodedSample:
        message = build_message(work, self._config)
        started = time.perf_counter()
        try:
            process_kwargs: Dict[str, Any] = {"return_video_kwargs": True}
            if self._config.model_type == "qwen3_vl":
                process_kwargs.update(image_patch_size=16, return_video_metadata=True)
            images, decoded_videos, video_kwargs = process_vision_info(
                message, **process_kwargs
            )
            if images is not None or decoded_videos is None or len(decoded_videos) != 1:
                raise ValueError("Expected exactly one decoded video and no images")

            if self._config.model_type == "qwen3_vl":
                video, video_metadata = decoded_videos[0]
                sampled_fps = None
            else:
                video = decoded_videos[0]
                video_metadata = None
                fps = video_kwargs.get("fps")
                if not isinstance(fps, list) or len(fps) != 1:
                    raise ValueError("Expected one sampled fps value")
                sampled_fps = float(fps[0])
        except Exception as error:
            raise _work_error(work, "video decode", error) from error

        return DecodedSample(
            sequence_id=work.sequence_id,
            record=work.record,
            message=message,
            video=video,
            video_metadata=video_metadata,
            sampled_fps=sampled_fps,
            decode_ms=(time.perf_counter() - started) * 1000,
        )


def prepare_batch(
    samples: Sequence[DecodedSample], processor: Any, config: PipelineConfig
) -> PreparedBatch:
    """Build one model batch; this is the only GPU batch-size boundary."""
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot prepare an empty Sports-QA batch")

    started = time.perf_counter()
    try:
        messages = [sample.message for sample in samples]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs: Dict[str, Any] = {
            "text": text,
            "images": None,
            "videos": [sample.video for sample in samples],
            "padding": True,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
        }
        if config.model_type == "qwen3_vl":
            processor_kwargs["video_metadata"] = [
                sample.video_metadata for sample in samples
            ]
        else:
            processor_kwargs["fps"] = [sample.sampled_fps for sample in samples]
        inputs = dict(processor(**processor_kwargs))
        attention_mask = inputs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise ValueError("Processor did not return a 2D attention_mask")
        input_token_counts = [
            int(value) for value in attention_mask.sum(dim=-1).tolist()
        ]
    except Exception as error:
        context = "; ".join(_sample_context(sample) for sample in samples)
        raise VideoPreparationError(
            f"Sports-QA processor preparation failed for [{context}]: {error}"
        ) from error

    if config.max_context is not None:
        for sample, input_token_count in zip(samples, input_token_counts):
            if input_token_count + config.max_new_tokens > config.max_context:
                error = RuntimeError(
                    "Video preprocessing produced "
                    f"{input_token_count} input tokens, which leaves insufficient context for "
                    f"{config.max_new_tokens} generated tokens "
                    f"(model limit: {config.max_context}). "
                    "Reduce --video-frames or --video-max-pixels."
                )
                work = QAWork(
                    sequence_id=sample.sequence_id,
                    video_path=Path(sample.message[0]["content"][0]["video"]),
                    record=sample.record,
                )
                raise _work_error(work, "context validation", error)

    return PreparedBatch(
        sequence_ids=[sample.sequence_id for sample in samples],
        records=[sample.record for sample in samples],
        inputs=inputs,
        input_token_counts=input_token_counts,
        decode_ms=sum(sample.decode_ms for sample in samples),
        processor_ms=(time.perf_counter() - started) * 1000,
    )


def initialize_worker(_: int, decoder_threads: int) -> None:
    """Keep decoder workers from oversubscribing CPU cores."""
    torch.set_num_threads(1)
    os.environ["TORCHCODEC_NUM_THREADS"] = str(decoder_threads)


def make_dataloader(
    dataset: SportsQADataset, decoder: SportsQADecoder, args: Any
) -> DataLoader:
    """Prefetch individual decoded QAs; GPU batching happens in the main process."""
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=decoder,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=False,
        timeout=args.timeout,
        multiprocessing_context=args.multiprocessing_context,
        worker_init_fn=partial(initialize_worker, decoder_threads=args.decoder_threads),
    )
