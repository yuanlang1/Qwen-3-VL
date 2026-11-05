"""Read selected original frames without importing a training or inference route."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import index_records, read_jsonl
from .records import parse_record
from .video_io import decode_frame_indices


def validate_selection(record: dict[str, Any]) -> None:
    parse_record(record)
    indices = record.get("selected_frame_indices")
    timestamps = record.get("selected_timestamps")
    kinds, clip_ids, scores = (record.get(key) for key in ("sampling_types", "clip_ids", "relevance_scores"))
    arrays = (indices, timestamps, kinds, clip_ids, scores)
    if any(not isinstance(value, list) for value in arrays) or not indices:
        raise ValueError("Selection requires non-empty frame indices and aligned arrays")
    if any(len(value) != len(indices) for value in arrays):
        raise ValueError("Selection arrays differ in length")
    if record.get("actual_frame_count") != len(indices):
        raise ValueError("actual_frame_count differs from selected frame count")
    total = record.get("source_video", {}).get("total_frames")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("Invalid source-video frame count")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total for index in indices):
        raise ValueError("Selected frame index is out of source-video bounds")
    if indices != sorted(set(indices)):
        raise ValueError("Selected frame indices must be sorted and unique")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in timestamps + scores):
        raise ValueError("Timestamps and relevance scores must be numbers")
    if not np.isfinite(timestamps).all() or not np.isfinite(scores).all():
        raise ValueError("Timestamps and relevance scores must be finite")
    if any(value < 0 for value in timestamps) or any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("Timestamps must be nonnegative and chronological")
    if any(kind not in {"slow", "fast"} for kind in kinds):
        raise ValueError("sampling_types must contain slow or fast")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < -1 for value in clip_ids):
        raise ValueError("clip_ids must be integers >= -1")
    config = record.get("selection", {})
    budget = config.get("frame_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0 or len(indices) > budget:
        raise ValueError("Selected frame count exceeds the configured budget")
    candidate_count = record.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or not 0 < candidate_count <= total:
        raise ValueError("Invalid candidate_count")
    if len(indices) != min(candidate_count, budget):
        raise ValueError("Selected frame count does not fill the available budget")
    if record.get("decoder") != "decord":
        raise ValueError("Selection must identify the source decoder as decord")


def load_selections(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Selection artifact does not exist: {path}")
    rows, _ = read_jsonl(path, recover_truncated_tail=False)
    selections = index_records(rows, path)
    for record in selections.values():
        validate_selection(record)
    return selections


def get_selection(records: dict[str, dict[str, Any]], sample_id: str | int) -> dict[str, Any]:
    return records[str(sample_id)]


def decode_selected_frames(video_path: str | Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Decode with the same backend used by selection and verify original times."""
    validate_selection(selection)
    frames, timestamps, total_frames = decode_frame_indices(Path(video_path), selection["selected_frame_indices"])
    if total_frames != selection["source_video"]["total_frames"]:
        raise ValueError("Source-video frame count has changed")
    if not np.allclose(timestamps, selection["selected_timestamps"], atol=0.001, rtol=0):
        raise ValueError("Selected timestamps do not match the source decoder")
    return {"frames": frames, "frame_indices": list(selection["selected_frame_indices"]), "timestamps": timestamps}
