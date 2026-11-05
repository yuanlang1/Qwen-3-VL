"""Dataset-independent sports QA records and explicit local-video resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpeg")


@dataclass(frozen=True)
class VideoQARecord:
    id: str
    video: str
    question: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON at {path}:{number}") from error
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects in a JSON array or JSONL file")
    return rows


def parse_record(row: dict[str, Any]) -> VideoQARecord:
    missing = {"id", "video", "question"}.difference(row)
    if missing:
        raise ValueError(f"Record is missing {sorted(missing)}")
    sample_id = row["id"]
    if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)) or not str(sample_id).strip():
        raise ValueError("id must be a non-empty string or integer")
    for name in ("video", "question"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"{name} must be a non-empty string for id={sample_id}")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object for id={sample_id}")
    return VideoQARecord(str(sample_id), row["video"].strip(), row["question"].strip(), metadata)


def load_records(path: Path, limit: int | None = None) -> list[VideoQARecord]:
    records = [parse_record(row) for row in read_rows(path)]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample id in {path}; use one input/output directory per dataset split")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    return records if limit is None else records[:limit]


class VideoResolver:
    """Resolve exact paths or explicit ID mappings without scanning sport directories."""

    def __init__(self, root: Path, index_path: Path | None = None):
        self.root = root.resolve()
        self.index = {}
        if index_path is not None:
            self.index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(self.index, dict) or any(not isinstance(value, str) for value in self.index.values()):
                raise ValueError("Video index must map video IDs to path strings")

    def resolve(self, video: str) -> Path:
        path = Path(self.index.get(video, video))
        path = path if path.is_absolute() else self.root / path
        if path.is_file():
            return path.resolve()
        if not path.suffix:
            matches = [path.with_suffix(extension) for extension in VIDEO_EXTENSIONS if path.with_suffix(extension).is_file()]
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                raise ValueError(f"Ambiguous video ID {video!r}; supply an exact extension or video index")
        raise FileNotFoundError(f"Cannot resolve video {video!r} below {self.root}")
