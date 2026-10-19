"""Read Sports-QA metadata and resolve metadata video IDs to extracted files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SportsQARecord:
    split: str
    qa_id: str
    video: str
    question: str

    @property
    def key(self) -> str:
        return f"{self.split}:{self.qa_id}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(metadata_root: Path, split: str, limit: int | None = None) -> list[SportsQARecord]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    path = metadata_root / f"{split}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Sports-QA metadata is missing: {path}") from error
    if not isinstance(payload, list):
        raise ValueError(f"Sports-QA metadata must be a JSON array: {path}")

    records: list[SportsQARecord] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} must be a JSON object")
        missing = {"qa_id", "video", "question"}.difference(row)
        if missing:
            raise ValueError(f"{path}:{index} is missing {sorted(missing)}")
        qa_id = str(row["qa_id"])
        if qa_id in seen_ids:
            raise ValueError(f"Duplicate qa_id in {path}: {qa_id}")
        seen_ids.add(qa_id)
        question = str(row["question"]).strip()
        if not question:
            raise ValueError(f"{path}:{index} has an empty question")
        records.append(SportsQARecord(split, qa_id, str(row["video"]), question))
        if limit is not None and len(records) >= limit:
            break
    return records


class VideoIndex:
    """Exact metadata-ID to video-file mapping with collision detection."""

    def __init__(self, video_root: Path) -> None:
        self.root = video_root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Sports-QA video root is not a directory: {self.root}")
        self._paths: dict[str, Path] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            key = path.relative_to(self.root).with_suffix("").as_posix()
            if key in self._paths:
                raise ValueError(
                    f"Video ID collision for {key}: {self._paths[key]} and {path}"
                )
            self._paths[key] = path

    @staticmethod
    def _key(video: str) -> str:
        return Path(video.replace("\\", "/")).with_suffix("").as_posix()

    def resolve(self, video: str) -> Path:
        direct = self.root / video
        if direct.is_file():
            return direct
        key = self._key(video)
        try:
            return self._paths[key]
        except KeyError as error:
            raise FileNotFoundError(f"Cannot resolve Sports-QA video ID: {video}") from error

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


def metadata_fingerprints(metadata_root: Path, splits: Iterable[str]) -> dict[str, str]:
    return {split: file_sha256(metadata_root / f"{split}.json") for split in splits}
