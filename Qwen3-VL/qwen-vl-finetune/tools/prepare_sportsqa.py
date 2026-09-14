"""Convert the official Sports-QA metadata into Qwen-VL training and eval files.

The source metadata stores video identifiers without extensions. This tool resolves
them against a server video root, or an optional JSON index, and never downloads
or copies videos.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_EXTENSIONS = (".mp4", ".avi", ".webm", ".mkv")
SPLITS = ("train", "val", "test")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def parse_extensions(values: Iterable[str]) -> Tuple[str, ...]:
    parsed = []
    for value in values:
        for extension in value.split(","):
            extension = extension.strip()
            if extension:
                parsed.append(extension if extension.startswith(".") else f".{extension}")
    return tuple(parsed) or DEFAULT_EXTENSIONS


def load_video_index(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    index = load_json(path)
    if not isinstance(index, dict):
        raise ValueError("--video-index must contain a JSON object mapping video IDs to paths")
    return {str(key): str(value) for key, value in index.items()}


def resolve_video(
    video_id: str,
    video_root: Path,
    extensions: Tuple[str, ...],
    supplied_index: Dict[str, str],
) -> Tuple[str, bool]:
    if video_id in supplied_index:
        value = supplied_index[video_id]
        path = Path(value)
        candidate = path if path.is_absolute() else video_root / path
        return value, candidate.is_file()

    raw_path = video_root / video_id
    if raw_path.is_file():
        return video_id, True

    for extension in extensions:
        candidate = video_root / f"{video_id}{extension}"
        if candidate.is_file():
            return f"{video_id}{extension}", True

    # Keep a deterministic candidate in --allow-missing mode so the generated
    # files can be inspected before videos arrive on the server.
    return f"{video_id}{extensions[0]}", False


def sport_from_video_id(video_id: str) -> str:
    return video_id.split("/", 1)[0]


def validate_row(row: Dict[str, Any], split: str) -> None:
    required = {"qa_id", "video", "type", "question", "answer", "ans_cls"}
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{split} row is missing fields: {sorted(missing)}")


def convert_row(row: Dict[str, Any], video_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    prompt = f"<video>\nQuestion: {question}\nAnswer with only the final answer."
    common = {
        "qa_id": row["qa_id"],
        "video_id": row["video"],
        "video": video_path,
        "type": row["type"],
        "sport": sport_from_video_id(str(row["video"])),
        "question": question,
        "answer": answer,
        "ans_cls": row["ans_cls"],
    }
    training_row = {
        "video": video_path,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        **common,
    }
    return training_row, common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Video extension to probe; repeat or pass a comma-separated list.",
    )
    parser.add_argument(
        "--video-index",
        type=Path,
        help="Optional JSON mapping metadata video IDs to paths relative to --video-root.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write deterministic placeholder paths instead of failing on missing videos.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Only convert the first N rows per split; intended for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = parse_extensions(args.extension)
    supplied_index = load_video_index(args.video_index)
    answer_to_id = load_json(args.metadata_root / "ans2cls.json")
    if not isinstance(answer_to_id, dict):
        raise ValueError("ans2cls.json must contain an answer-to-class mapping")

    all_video_index: Dict[str, str] = {}
    missing_video_ids = set()
    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split in SPLITS:
        rows = load_json(args.metadata_root / f"{split}.json")
        if not isinstance(rows, list):
            raise ValueError(f"{split}.json must contain a JSON list")
        if args.limit_per_split is not None:
            rows = rows[: args.limit_per_split]

        training_rows: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []
        type_counts: Counter[str] = Counter()

        for row in rows:
            validate_row(row, split)
            video_id = str(row["video"])
            video_path, exists = resolve_video(
                video_id, args.video_root, extensions, supplied_index
            )
            all_video_index[video_id] = video_path
            if not exists:
                missing_video_ids.add(video_id)
            training_row, manifest_row = convert_row(row, video_path)
            training_rows.append(training_row)
            manifest_rows.append(manifest_row)
            type_counts[str(row["type"])] += 1

        write_json(args.output_dir / f"{split}.json", training_rows)
        write_json(args.output_dir / f"{split}_manifest.json", manifest_rows)
        split_summaries[split] = {
            "qa_count": len(rows),
            "unique_video_count": len({str(row["video"]) for row in rows}),
            "question_type_counts": dict(sorted(type_counts.items())),
        }

    write_json(args.output_dir / "answer_to_id.json", answer_to_id)
    write_json(
        args.output_dir / "id_to_answer.json",
        {str(value): key for key, value in answer_to_id.items()},
    )
    write_json(args.output_dir / "video_index.json", all_video_index)
    write_json(args.output_dir / "missing_videos.json", sorted(missing_video_ids))
    summary = {
        "metadata_root": str(args.metadata_root),
        "video_root": str(args.video_root),
        "extensions": list(extensions),
        "answer_class_count": len(answer_to_id),
        "missing_video_count": len(missing_video_ids),
        "splits": split_summaries,
    }
    write_json(args.output_dir / "preparation_summary.json", summary)

    if missing_video_ids and not args.allow_missing:
        raise SystemExit(
            f"{len(missing_video_ids)} videos are missing. See "
            f"{args.output_dir / 'missing_videos.json'} or pass --allow-missing for a dry run."
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
