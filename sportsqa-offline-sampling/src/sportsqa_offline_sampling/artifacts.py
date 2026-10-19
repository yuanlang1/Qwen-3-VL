"""Versioned JSONL artifacts with safe resume behavior."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ConfigMismatchError(RuntimeError):
    """Raised before incompatible artifacts are read or appended to."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path, recover_truncated_tail: bool = True) -> tuple[list[dict[str, Any]], int]:
    """Return records and the valid byte length; only a final partial line is recoverable."""
    if not path.exists():
        return [], 0
    raw = path.read_bytes()
    if not raw:
        return [], 0
    records: list[dict[str, Any]] = []
    valid_end = 0
    chunks = raw.splitlines(keepends=True)
    for index, chunk in enumerate(chunks):
        end = valid_end + len(chunk)
        content = chunk.rstrip(b"\r\n")
        if not content:
            valid_end = end
            continue
        try:
            record = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            is_last = index == len(chunks) - 1
            has_newline = chunk.endswith((b"\n", b"\r"))
            if recover_truncated_tail and is_last and not has_newline:
                return records, valid_end
            raise ValueError(f"Invalid JSONL record in {path} at line {index + 1}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path} at line {index + 1}")
        records.append(record)
        valid_end = end
    return records, valid_end


def prepare_jsonl_for_append(path: Path) -> list[dict[str, Any]]:
    records, valid_end = read_jsonl(path)
    if path.exists() and path.stat().st_size != valid_end:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
    return records


def index_records(records: Iterable[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            key = f"{record['split']}:{record['qa_id']}"
        except KeyError as error:
            raise ValueError(f"Artifact record in {path} lacks split or qa_id") from error
        if key in indexed:
            raise ValueError(f"Duplicate artifact record in {path}: {key}")
        indexed[key] = record
    return indexed


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(canonical_json(record) + "\n")
    handle.flush()


@dataclass(frozen=True)
class StageArtifact:
    output_path: Path
    config_path: Path
    config: dict[str, Any]

    def prepare(self, overwrite: bool = False) -> None:
        expected = {"fingerprint": fingerprint(self.config), "config": self.config}
        if self.config_path.exists():
            current = json.loads(self.config_path.read_text(encoding="utf-8"))
            if current != expected:
                if not overwrite:
                    raise ConfigMismatchError(
                        f"Configuration mismatch for {self.output_path}. "
                        "Use --overwrite to create a separate run directory or replace this artifact."
                    )
                if self.output_path.exists():
                    self.output_path.unlink()
                self.config_path.unlink()
        elif self.output_path.exists() and not overwrite:
            raise ConfigMismatchError(
                f"{self.output_path} exists without its configuration record; refusing to append."
            )
        if overwrite and self.output_path.exists():
            self.output_path.unlink()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class FailureRecorder:
    def __init__(self, path: Path, config_fingerprint: str) -> None:
        self.path = path
        self.config_fingerprint = config_fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, split: str, qa_id: str, video: str, error: Exception) -> None:
        record = {
            "split": split,
            "qa_id": str(qa_id),
            "video": video,
            "config_fingerprint": self.config_fingerprint,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            append_jsonl(handle, record)
