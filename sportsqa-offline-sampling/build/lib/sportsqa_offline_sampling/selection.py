"""Deterministic Sports-TCS Slow/Fast frame selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Selection:
    frame_indices: list[int]
    sampling_types: list[str]
    clip_ids: list[int]
    relevance_scores: list[float]
    effective_alpha: float
    clips: list[tuple[int, int]]


def adaptive_candidate_fps(
    duration_seconds: float,
    short_video_seconds: float = 30.0,
    medium_video_seconds: float = 120.0,
) -> float:
    if short_video_seconds <= 0 or medium_video_seconds <= short_video_seconds:
        raise ValueError("Require 0 < short_video_seconds < medium_video_seconds")
    if duration_seconds <= short_video_seconds:
        return 4.0
    if duration_seconds <= medium_video_seconds:
        return 2.0
    return 1.0


def candidate_frame_indices(total_frames: int, source_fps: float, candidate_fps: float) -> list[int]:
    if total_frames <= 0 or source_fps <= 0 or candidate_fps <= 0:
        raise ValueError("total_frames, source_fps, and candidate_fps must be positive")
    duration_seconds = total_frames / source_fps
    timestamps = np.arange(0.0, duration_seconds, 1.0 / candidate_fps)
    indices = np.round(timestamps * source_fps).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return np.unique(np.concatenate((indices, np.asarray([total_frames - 1])))).tolist()


def _smooth(scores: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    if scores.size < 2 or radius <= 0:
        return scores.copy()
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(np.pad(scores, (radius, radius), mode="edge"), kernel, mode="valid")


def _find_clips(scores: np.ndarray, alpha: float) -> list[tuple[int, int]]:
    threshold = float(scores.mean() + alpha * scores.std())
    peaks = [
        index
        for index, value in enumerate(scores)
        if value >= threshold
        and (index == 0 or value >= scores[index - 1])
        and (index == len(scores) - 1 or value >= scores[index + 1])
    ]
    clips: list[tuple[int, int]] = []
    for peak in peaks:
        left = peak
        while left > 0 and threshold <= scores[left - 1] <= scores[left]:
            left -= 1
        right = peak
        while right + 1 < len(scores) and threshold <= scores[right + 1] <= scores[right]:
            right += 1
        clips.append((left, right))
    merged: list[tuple[int, int]] = []
    for left, right in sorted(clips):
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _uniform_pick(values: Sequence[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    ordered = sorted(values)
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[position] for position in positions]


def select_slow_fast_frames(
    query_frame_scores: Sequence[Sequence[float]],
    candidate_indices: Sequence[int],
    *,
    budget: int = 16,
    alpha: float = 0.5,
    gaussian_radius: int = 4,
    gaussian_sigma: float = 1.0,
    fast_ratio: float = 0.25,
) -> Selection:
    """Select ordered frames from question-conditioned candidate-frame scores."""
    scores = np.asarray(query_frame_scores, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != candidates.size:
        raise ValueError("Scores must have shape [queries, len(candidate_indices)]")
    if candidates.size == 0 or budget <= 0 or not 0 <= fast_ratio <= 1:
        raise ValueError("Selection requires candidates, a positive budget, and fast_ratio in [0, 1]")
    if np.any(np.diff(candidates) <= 0):
        raise ValueError("candidate_indices must be strictly increasing")

    budget = min(int(budget), int(candidates.size))
    fast_count = int(budget * fast_ratio)
    slow_count = budget - fast_count
    relevance = _smooth(scores.mean(axis=0), gaussian_radius, gaussian_sigma)

    current_alpha = float(alpha)
    clips = _find_clips(relevance, current_alpha)
    best = (clips, current_alpha, float("inf"))
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(16):
        clip_positions = {position for left, right in clips for position in range(left, right + 1)}
        shortage = max(0, slow_count - len(clip_positions)) + max(
            0, fast_count - (candidates.size - len(clip_positions))
        )
        if shortage < best[2]:
            best = (clips, current_alpha, shortage)
        partition = tuple(clips)
        if partition in seen:
            break
        seen.add(partition)
        if len(clip_positions) < slow_count:
            current_alpha *= 0.5
        elif candidates.size - len(clip_positions) < fast_count:
            current_alpha *= 2.0
        else:
            break
        clips = _find_clips(relevance, current_alpha)
    else:
        clips, current_alpha = best[0], best[1]

    clip_positions = {position for left, right in clips for position in range(left, right + 1)}
    if len(clip_positions) < slow_count or candidates.size - len(clip_positions) < fast_count:
        clips, current_alpha = best[0], best[1]
        clip_positions = {position for left, right in clips for position in range(left, right + 1)}

    clip_id_by_position = {
        position: clip_id
        for clip_id, (left, right) in enumerate(clips)
        for position in range(left, right + 1)
    }
    slow_positions = sorted(clip_id_by_position)
    fast_positions = [position for position in range(candidates.size) if position not in clip_id_by_position]

    if len(slow_positions) < slow_count:
        supplemental = sorted(fast_positions, key=lambda item: relevance[item], reverse=True)[: slow_count - len(slow_positions)]
        slow_positions = sorted(slow_positions + supplemental)
        fast_positions = [position for position in fast_positions if position not in set(supplemental)]
    if len(fast_positions) < fast_count:
        supplemental = sorted(slow_positions, key=lambda item: relevance[item])[: fast_count - len(fast_positions)]
        fast_positions = sorted(fast_positions + supplemental)
        slow_positions = [position for position in slow_positions if position not in set(supplemental)]

    chosen = [(position, "slow") for position in _uniform_pick(slow_positions, slow_count)]
    chosen.extend((position, "fast") for position in _uniform_pick(fast_positions, fast_count))
    used = {position for position, _ in chosen}
    if len(chosen) < budget:
        for position in _uniform_pick([item for item in range(candidates.size) if item not in used], budget - len(chosen)):
            chosen.append((position, "slow" if position in clip_id_by_position else "fast"))

    chosen.sort(key=lambda item: candidates[item[0]])
    return Selection(
        frame_indices=[int(candidates[position]) for position, _ in chosen],
        sampling_types=[kind for _, kind in chosen],
        clip_ids=[clip_id_by_position.get(position, -1) if kind == "slow" else -1 for position, kind in chosen],
        relevance_scores=[float(relevance[position]) for position, _ in chosen],
        effective_alpha=current_alpha,
        clips=clips,
    )
