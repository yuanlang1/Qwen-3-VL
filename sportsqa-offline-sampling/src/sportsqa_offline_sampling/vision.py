"""Video inspection, preview extraction, and cached OpenCLIP scoring."""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import canonical_json, fingerprint
from .selection import candidate_frame_indices


@dataclass(frozen=True)
class VideoInfo:
    total_frames: int
    average_fps: float

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.average_fps


def _video_reader(path: Path):
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise ImportError("Video processing requires decord. Install requirements.txt.") from error
    return VideoReader(str(path), ctx=cpu(0), num_threads=1)


def inspect_video(path: Path) -> VideoInfo:
    reader = _video_reader(path)
    try:
        total_frames = len(reader)
        average_fps = float(reader.get_avg_fps())
    finally:
        reader.seek(0)
    if total_frames <= 0 or average_fps <= 0:
        raise ValueError(f"Cannot read valid video metadata: {path}")
    return VideoInfo(total_frames=total_frames, average_fps=average_fps)


def read_frame_timestamps(path: Path, indices: Sequence[int]) -> list[float]:
    if not indices:
        return []
    reader = _video_reader(path)
    try:
        values = np.asarray(reader.get_frame_timestamp(list(indices)), dtype=np.float64)
    finally:
        reader.seek(0)
    if values.ndim == 2:
        values = values[:, 0]
    if values.ndim != 1 or values.size != len(indices):
        raise ValueError(f"Decoder returned invalid timestamps for {path}")
    return [float(value) for value in values]


def decode_preview_frames(path: Path, count: int = 4, max_resolution: int = 224):
    """Decode at most ``count`` uniformly spaced low-resolution preview frames."""
    if count <= 0 or max_resolution <= 0:
        raise ValueError("count and max_resolution must be positive")
    reader = _video_reader(path)
    try:
        total_frames = len(reader)
        if total_frames <= 0:
            raise ValueError(f"Cannot decode preview from an empty video: {path}")
        indices = np.linspace(0, total_frames - 1, min(count, total_frames), dtype=int).tolist()
        arrays = reader.get_batch(indices).asnumpy()
    finally:
        reader.seek(0)
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Preview extraction requires Pillow. Install requirements.txt.") from error
    resampling = getattr(Image, "Resampling", Image)
    images = []
    for array in arrays:
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail((max_resolution, max_resolution), resampling.LANCZOS)
        images.append(image)
    return images, indices


class OpenCLIPFrameScorer:
    """Lazy OpenCLIP scorer; neither OpenCLIP nor Torch is imported for non-score stages."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
        image_batch_size: int = 64,
        text_batch_size: int = 256,
    ) -> None:
        try:
            import open_clip
            import torch
            import torch.nn.functional as functional
        except ImportError as error:
            raise ImportError("CLIP scoring requires torch and open_clip_torch.") from error
        if image_batch_size <= 0 or text_batch_size <= 0:
            raise ValueError("image_batch_size and text_batch_size must be positive")
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CLIP scoring but is unavailable")
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.model_name = model_name
        self.pretrained = pretrained
        precision = "fp16" if self.device.type == "cuda" else "fp32"
        self.model, _, self.image_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device, precision=precision
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_video_frames(self, video_path: Path, indices: Sequence[int]):
        if not indices:
            raise ValueError("At least one candidate frame is required")
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("CLIP scoring requires Pillow.") from error
        reader = _video_reader(video_path)
        feature_batches = []
        try:
            with self.torch.inference_mode():
                for start in range(0, len(indices), self.image_batch_size):
                    batch_indices = list(indices[start : start + self.image_batch_size])
                    arrays = reader.get_batch(batch_indices).asnumpy()
                    images = [Image.fromarray(array).convert("RGB") for array in arrays]
                    pixels = self.torch.stack([self.image_preprocess(image) for image in images]).to(self.device)
                    features = self.model.encode_image(pixels)
                    feature_batches.append(self.functional.normalize(features.float(), dim=-1).cpu())
        finally:
            reader.seek(0)
        return self.torch.cat(feature_batches, dim=0)

    def score_queries(self, queries: Sequence[str], image_features):
        if not queries:
            raise ValueError("At least one query is required")
        text_batches = []
        with self.torch.inference_mode():
            for start in range(0, len(queries), self.text_batch_size):
                tokens = self.tokenizer(list(queries[start : start + self.text_batch_size])).to(self.device)
                features = self.model.encode_text(tokens)
                text_batches.append(self.functional.normalize(features.float(), dim=-1).cpu())
        return self.torch.cat(text_batches, dim=0) @ image_features.T


class FeatureCache:
    """Persist candidate-frame CLIP features separately from per-QA text scores."""

    def __init__(self, root: Path, config: dict[str, object]) -> None:
        self.root = root / fingerprint(config)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _video_fingerprint(video_path: Path) -> dict[str, int]:
        stat = video_path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _path(self, video_id: str) -> Path:
        return self.root / f"{fingerprint({'video': video_id})}.npz"

    def load(self, video_id: str, video_path: Path, expected: dict[str, object]):
        path = self._path(video_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            expected_metadata = expected | {"video_stat": self._video_fingerprint(video_path)}
            if metadata != expected_metadata:
                raise RuntimeError(
                    f"Cached CLIP features do not match the requested video/configuration: {path}. "
                    "Use a new output directory after changing source videos or settings."
                )
            return np.asarray(data["features"], dtype=np.float32)

    def save(self, video_id: str, video_path: Path, expected: dict[str, object], features: np.ndarray) -> None:
        path = self._path(video_id)
        metadata = expected | {"video_stat": self._video_fingerprint(video_path)}
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=canonical_json(metadata), features=np.asarray(features, dtype=np.float32))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


def candidate_descriptor(info: VideoInfo, candidate_fps: float, indices: Sequence[int], timestamps: Sequence[float]) -> dict[str, object]:
    return {
        "fps": candidate_fps,
        "frame_indices": [int(index) for index in indices],
        "timestamps": [float(timestamp) for timestamp in timestamps],
        "source": asdict(info),
    }


def build_candidates(info: VideoInfo, candidate_fps: float) -> list[int]:
    return candidate_frame_indices(info.total_frames, info.average_fps, candidate_fps)
