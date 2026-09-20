"""Validate a prepared Sports-QA annotation file before training.

The validator never changes the source annotation. It writes an ordered valid
annotation plus an auditable JSONL quarantine file for records whose video is
missing or, when requested, fails an ffprobe stream check.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("--annotation must contain a JSON list")
    return records


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def resolve_video_path(video_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else video_root / path


def video_error(
    path: Path, verify_video: bool, ffprobe_bin: str, timeout: float
) -> Optional[Tuple[str, str]]:
    if not path.is_file():
        return "missing_video", f"Video file does not exist: {path}"
    if not verify_video:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "video_probe_timeout", f"ffprobe exceeded {timeout:g}s: {path}"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown ffprobe error"
        return "video_probe_error", detail
    if "video" not in result.stdout.split():
        return "no_video_stream", "ffprobe found no video stream"
    return None


def quarantine_record(
    index: int,
    record: Any,
    error_kind: str,
    error: str,
    video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    source = record if isinstance(record, dict) else {}
    return {
        "source_index": index,
        "qa_id": source.get("qa_id"),
        "video": source.get("video"),
        "video_path": str(video_path) if video_path is not None else None,
        "error_kind": error_kind,
        "error": error,
    }


def validate_records(
    records: List[Dict[str, Any]],
    video_root: Path,
    verify_video: bool,
    ffprobe_bin: str,
    timeout: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter[str], int]:
    valid_records = []
    quarantined = []
    failure_counts: Counter[str] = Counter()
    checked_videos: Dict[Path, Optional[Tuple[str, str]]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            error_kind = "invalid_record"
            error = "Expected a JSON object"
            quarantined.append(quarantine_record(index, record, error_kind, error))
            failure_counts[error_kind] += 1
            continue

        missing_fields = [
            field
            for field in ("qa_id", "video", "conversations")
            if field not in record or record[field] in (None, "")
        ]
        if missing_fields:
            error_kind = "missing_required_field"
            error = f"Missing required fields: {', '.join(missing_fields)}"
            quarantined.append(quarantine_record(index, record, error_kind, error))
            failure_counts[error_kind] += 1
            continue
        if not isinstance(record["video"], str):
            error_kind = "invalid_video_path"
            error = "The video field must be a string"
            quarantined.append(quarantine_record(index, record, error_kind, error))
            failure_counts[error_kind] += 1
            continue

        path = resolve_video_path(video_root, record["video"])
        cached_error = checked_videos.get(path)
        if path not in checked_videos:
            cached_error = video_error(path, verify_video, ffprobe_bin, timeout)
            checked_videos[path] = cached_error
        if cached_error is not None:
            error_kind, error = cached_error
            quarantined.append(
                quarantine_record(index, record, error_kind, error, video_path=path)
            )
            failure_counts[error_kind] += 1
            continue
        valid_records.append(record)

    return valid_records, quarantined, failure_counts, len(checked_videos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-video",
        action="store_true",
        help="Use ffprobe to require a readable video stream, not only file existence.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--video-timeout", type=float, default=30.0)
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Write all reports, then exit nonzero if any record was quarantined.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_timeout <= 0:
        raise ValueError("--video-timeout must be greater than zero")
    if args.verify_video and shutil.which(args.ffprobe_bin) is None:
        raise RuntimeError(
            f"--verify-video requires an ffprobe executable: {args.ffprobe_bin!r}"
        )

    records = load_records(args.annotation)
    valid_records, quarantined, failure_counts, checked_video_count = validate_records(
        records,
        args.video_root,
        args.verify_video,
        args.ffprobe_bin,
        args.video_timeout,
    )
    stem = args.annotation.stem
    valid_path = args.output_dir / f"valid_{stem}.json"
    quarantine_path = args.output_dir / f"quarantined_{stem}.jsonl"
    summary_path = args.output_dir / f"validation_summary_{stem}.json"
    summary = {
        "annotation": str(args.annotation),
        "video_root": str(args.video_root),
        "verify_video": args.verify_video,
        "input_record_count": len(records),
        "valid_record_count": len(valid_records),
        "quarantined_record_count": len(quarantined),
        "checked_video_count": checked_video_count,
        "failure_counts": dict(sorted(failure_counts.items())),
        "valid_annotation": str(valid_path),
        "quarantine_file": str(quarantine_path),
    }
    atomic_write(valid_path, json.dumps(valid_records, ensure_ascii=False, indent=2) + "\n")
    atomic_write(
        quarantine_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantined),
    )
    atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if quarantined and args.fail_on_invalid:
        raise SystemExit(f"Quarantined {len(quarantined)} invalid record(s)")


if __name__ == "__main__":
    main()
