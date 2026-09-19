"""Run deterministic Qwen-VL inference on a prepared Sports-QA manifest."""

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

from sportsqa_dataloader import (
    DecodedSample,
    PipelineConfig,
    PreparedBatch,
    SportsQADecoder,
    SportsQADataset,
    build_qa_work_items,
    configure_video_processor,
    make_dataloader,
    move_inputs_to_device,
    prepare_batch,
)


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


def select_pending_records(
    records: List[Dict[str, Any]], completed_ids: set[str], limit: Optional[int],
) -> List[Dict[str, Any]]:
    pending = [
        record for record in records if str(record["qa_id"]) not in completed_ids
    ]
    return pending if limit is None else pending[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="GPU model.generate batch size; DataLoader tasks always contain one QA.",
    )
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
        "--video-backend",
        choices=("torchcodec", "decord", "torchvision", "auto"),
        default="torchcodec",
        help=(
            "CPU video decoder. The default fails early if TorchCodec cannot load; "
            "use auto only when fallback is intentional."
        ),
    )
    parser.add_argument(
        "--decoder-threads", type=int, default=2, help="FFmpeg threads per worker."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="CPU video-decode workers."
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Individual decoded QAs prefetched per worker.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep CPU decode workers alive until the DataLoader is destroyed.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin processor outputs in the main process before asynchronous H2D.",
    )
    parser.add_argument(
        "--use-fast-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select the Transformers fast image/video processor explicitly.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Maximum seconds to wait for the next decoded QA.",
    )
    parser.add_argument(
        "--multiprocessing-context",
        default="spawn",
        help="PyTorch DataLoader multiprocessing start method.",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.video_frames < 2 or args.video_frames % 2:
        raise ValueError("--video-frames must be an even integer of at least 2")
    if args.video_min_pixels < 1 or args.video_max_pixels < args.video_min_pixels:
        raise ValueError(
            "Video pixel limits must satisfy 1 <= min_pixels <= max_pixels"
        )
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1 for the DataLoader pipeline")
    if args.decoder_threads < 1:
        raise ValueError("--decoder-threads must be at least 1")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")


VIDEO_BACKEND_MODULES = {
    "torchcodec": "torchcodec.decoders",
    "decord": "decord",
    "torchvision": "torchvision",
}


def configure_video_backend(requested: str) -> str:
    """Select a decoder before spawning workers and verify that it can load."""
    candidates = list(VIDEO_BACKEND_MODULES) if requested == "auto" else [requested]
    failures = []
    for backend in candidates:
        try:
            importlib.import_module(VIDEO_BACKEND_MODULES[backend])
        except Exception as error:
            failures.append(f"{backend}: {error}")
            continue
        os.environ["FORCE_QWENVL_VIDEO_READER"] = backend
        vision_process = importlib.import_module("qwen_vl_utils.vision_process")
        if hasattr(vision_process, "FORCE_QWENVL_VIDEO_READER"):
            vision_process.FORCE_QWENVL_VIDEO_READER = backend
        backend_selector = getattr(vision_process, "get_video_reader_backend", None)
        if hasattr(backend_selector, "cache_clear"):
            backend_selector.cache_clear()
        return backend

    details = "; ".join(failures)
    if requested == "torchcodec":
        raise RuntimeError(
            "--video-backend=torchcodec was requested, but TorchCodec could not load. "
            "This project expects torchcodec==0.7.0 with torch==2.8.0 and a shared "
            "FFmpeg 4-9 installation visible to the Python process. Verify with "
            '`python -c "from torchcodec.decoders import VideoDecoder"` and '
            "`ffmpeg -version`. Original error: "
            f"{details}"
        )
    raise RuntimeError(
        f"No usable video backend was found for --video-backend={requested}: {details}"
    )


def iter_gpu_batches(
    samples: Iterable[DecodedSample], batch_size: int
) -> Iterator[Tuple[List[DecodedSample], float]]:
    """Group ordered decoded samples; batch_size has no effect inside DataLoader."""
    iterator = iter(samples)
    while True:
        batch = []
        loader_wait_ms = 0.0
        for _ in range(batch_size):
            started = time.perf_counter()
            try:
                sample = next(iterator)
            except StopIteration:
                if batch:
                    yield batch, loader_wait_ms
                return
            loader_wait_ms += (time.perf_counter() - started) * 1000
            batch.append(sample)
        yield batch, loader_wait_ms


@dataclass
class InferenceTimings:
    decode_ms: float = 0.0
    processor_ms: float = 0.0
    loader_wait_ms: float = 0.0
    pin_memory_ms: float = 0.0
    h2d_enqueue_ms: float = 0.0
    generate_ms: float = 0.0
    batch_count: int = 0

    def add(
        self,
        batch: PreparedBatch,
        loader_wait_ms: float,
        pin_memory_ms: float,
        h2d_enqueue_ms: float,
        generate_ms: float,
    ) -> None:
        self.decode_ms += batch.decode_ms
        self.processor_ms += batch.processor_ms
        self.loader_wait_ms += loader_wait_ms
        self.pin_memory_ms += pin_memory_ms
        self.h2d_enqueue_ms += h2d_enqueue_ms
        self.generate_ms += generate_ms
        self.batch_count += 1

    def print_summary(self) -> None:
        print(
            "Sports-QA pipeline timing (ms): "
            f"decode={self.decode_ms:.1f}, processor={self.processor_ms:.1f}, "
            f"dataloader_wait={self.loader_wait_ms:.1f}, "
            f"pin_memory={self.pin_memory_ms:.1f}, "
            f"h2d_enqueue={self.h2d_enqueue_ms:.1f}, generate={self.generate_ms:.1f}, "
            f"batches={self.batch_count}"
        )


def max_context_length(model: Any) -> Optional[int]:
    max_context = getattr(model.config, "max_position_embeddings", None)
    if max_context is None:
        max_context = getattr(
            getattr(model.config, "text_config", None), "max_position_embeddings", None
        )
    return max_context


def generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    return kwargs


def decode_predictions(
    processor: Any, inputs: Dict[str, Any], generated_ids: Any
) -> List[str]:
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


def write_predictions(
    output_handle: Any,
    batch: PreparedBatch,
    predictions: List[str],
    args: argparse.Namespace,
) -> None:
    if len(predictions) != len(batch.records):
        raise RuntimeError(
            f"Model returned {len(predictions)} predictions for {len(batch.records)} QA records"
        )
    for record, prediction in zip(batch.records, predictions):
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


def main() -> None:
    args = parse_args()
    validate_args(args)

    records = load_json(args.manifest)
    if not isinstance(records, list):
        raise ValueError("--manifest must contain a JSON list")
    completed_ids = read_completed_ids(args.output_file) if args.resume else set()
    records = select_pending_records(records, completed_ids, args.limit)
    if not records:
        print(
            "Sports-QA inference: no pending QAs; existing predictions are unchanged."
        )
        return
    work_items = build_qa_work_items(records, args.video_root)
    os.environ["TORCHCODEC_NUM_THREADS"] = str(args.decoder_threads)
    video_backend = configure_video_backend(args.video_backend)

    model_kwargs: Dict[str, Any] = {"dtype": args.dtype, "device_map": args.device_map}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        use_fast=args.use_fast_processor,
    )

    pipeline_config = PipelineConfig(
        model_type=model.config.model_type,
        prompt_style=args.prompt_style,
        video_frames=args.video_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        max_new_tokens=args.max_new_tokens,
        max_context=max_context_length(model),
    )
    configure_video_processor(processor, pipeline_config)
    dataset = SportsQADataset(work_items)
    decoder = SportsQADecoder(pipeline_config)
    dataloader = make_dataloader(dataset, decoder, args)
    print(
        f"Video sampling: model_type={model.config.model_type}, frames={args.video_frames}; "
        f"prompt_style={args.prompt_style}; {len(records)} independent QAs; "
        f"gpu_batch_size={args.batch_size}; video_backend={video_backend}, "
        f"decoder_threads={args.decoder_threads}; num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor} samples/worker"
    )

    timings = InferenceTimings()
    kwargs = generation_kwargs(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output_file.open(mode, encoding="utf-8") as output_handle:
        with tqdm(
            total=len(records),
            desc="Sports-QA inference",
            unit="qa",
            dynamic_ncols=True,
        ) as progress:
            for samples, loader_wait_ms in iter_gpu_batches(
                dataloader, args.batch_size
            ):
                batch = prepare_batch(samples, processor, pipeline_config)

                pin_started = time.perf_counter()
                if args.pin_memory and torch.cuda.is_available():
                    batch.pin_memory()
                pin_memory_ms = (time.perf_counter() - pin_started) * 1000

                h2d_started = time.perf_counter()
                inputs = move_inputs_to_device(batch.inputs, model.device)
                h2d_enqueue_ms = (time.perf_counter() - h2d_started) * 1000

                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **kwargs)
                generate_ms = (time.perf_counter() - generation_started) * 1000

                predictions = decode_predictions(processor, inputs, generated_ids)
                write_predictions(output_handle, batch, predictions, args)
                timings.add(
                    batch, loader_wait_ms, pin_memory_ms, h2d_enqueue_ms, generate_ms,
                )
                progress.update(len(batch.records))
                progress.set_postfix(tokens=max(batch.input_token_counts))
    timings.print_summary()


if __name__ == "__main__":
    main()
