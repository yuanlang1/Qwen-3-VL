"""Decord video inspection, previews, and exact original-frame decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np



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
        raise ImportError("Video processing requires decord. Install sports-video-sampling[video].") from error
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
        raise ImportError("Preview extraction requires Pillow. Install sports-video-sampling[video].") from error
    resampling = getattr(Image, "Resampling", Image)
    images = []
    for array in arrays:
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail((max_resolution, max_resolution), resampling.LANCZOS)
        images.append(image)
    return images, indices


def decode_frame_indices(path: Path, indices: Sequence[int]):
    """Return RGB uint8 frames [T,H,W,C] and decoder-origin timestamps."""
    if not indices:
        raise ValueError("At least one frame index is required")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("Frame indices must be integers")
    if list(indices) != sorted(set(indices)):
        raise ValueError("Frame indices must be sorted and unique")
    reader = _video_reader(path)
    try:
        if indices[0] < 0 or indices[-1] >= len(reader):
            raise ValueError("Frame index is outside the source video")
        frames = reader.get_batch(list(indices)).asnumpy()
        timestamps = np.asarray(reader.get_frame_timestamp(list(indices)), dtype=np.float64)
        if timestamps.ndim == 2:
            timestamps = timestamps[:, 0]
        return frames, timestamps.tolist(), len(reader)
    finally:
        reader.seek(0)


