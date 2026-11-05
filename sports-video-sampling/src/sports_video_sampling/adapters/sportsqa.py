"""Sports-QA field conversion; answers never become retrieval context."""

from __future__ import annotations

from typing import Any


def convert_sportsqa(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["qa_id"],
        "video": row["video"],
        "question": row["question"],
        "metadata": {"dataset": "sportsqa", **row.get("metadata", {})},
    }
