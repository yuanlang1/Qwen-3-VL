"""Convert one-video sports QA conversations; reject ambiguous multi-turn input."""

from __future__ import annotations

import re
from typing import Any


def convert_qwen(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = next((row[key] for key in ("id", "qa_id", "question_id", "sample_id") if key in row), None)
    if sample_id is None:
        raise ValueError("Qwen record requires id, qa_id, question_id, or sample_id")
    video = row.get("video")
    if isinstance(video, list) and len(video) == 1:
        video = video[0]
    if not isinstance(video, str):
        raise ValueError(f"Qwen record {sample_id} requires exactly one video")
    if row.get("image"):
        raise ValueError(f"Qwen record {sample_id} mixes image and video inputs")
    turns = [turn for turn in row.get("conversations", []) if turn.get("from") == "human"]
    if len(turns) != 1:
        raise ValueError(f"Qwen record {sample_id} requires exactly one human question")
    text = turns[0].get("value")
    if not isinstance(text, str) or text.count("<video>") > 1 or "<image>" in text:
        raise ValueError(f"Qwen record {sample_id} has ambiguous media placeholders")
    return {
        "id": sample_id,
        "video": video,
        "question": re.sub(r"<video>", "", text).strip(),
        "metadata": row.get("metadata", {}),
    }
