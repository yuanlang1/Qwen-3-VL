"""Score Sports-QA predictions against the official answer vocabulary."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                rows.append(row)
    return rows


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"^```(?:text)?", "", text.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    text = re.sub(
        r"^(?:final\s+answer|the\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip(" \t\n\r.,;:!\"'")


def calculate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"count": 0, "accuracy": None, "macro_f1": None}

    labels = sorted({int(row["ans_cls"]) for row in rows})
    correct = sum(int(row["correct"]) for row in rows)
    f1_scores = []
    per_class = {}
    for label in labels:
        true_positive = sum(
            int(row["ans_cls"]) == label and row["prediction_cls"] == label for row in rows
        )
        false_positive = sum(
            int(row["ans_cls"]) != label and row["prediction_cls"] == label for row in rows
        )
        false_negative = sum(
            int(row["ans_cls"]) == label and row["prediction_cls"] != label for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
        f1_scores.append(f1)
        per_class[str(label)] = {
            "support": true_positive + false_negative,
            "f1": f1,
        }
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
    }


def group_metrics(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {group: calculate_metrics(group_rows) for group, group_rows in sorted(groups.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--answer-map", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--details-file", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only score the first N manifest records; intended for smoke tests.",
    )
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Score absent predictions as incorrect instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    answer_map = load_json(args.answer_map)
    if not isinstance(manifest, list) or not isinstance(answer_map, dict):
        raise ValueError("Manifest must be a list and answer map must be an object")
    if args.limit is not None:
        manifest = manifest[: args.limit]

    normalized_to_class: Dict[str, int] = {}
    for answer, class_id in answer_map.items():
        normalized = normalize_answer(answer)
        if normalized in normalized_to_class:
            raise ValueError(f"Ambiguous normalized answer label: {normalized!r}")
        normalized_to_class[normalized] = int(class_id)

    prediction_by_id: Dict[str, Dict[str, Any]] = {}
    for prediction in load_jsonl(args.predictions):
        qa_id = str(prediction["qa_id"])
        if qa_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction for qa_id={qa_id}")
        prediction_by_id[qa_id] = prediction

    scored_rows: List[Dict[str, Any]] = []
    missing_ids = []
    for item in manifest:
        qa_id = str(item["qa_id"])
        prediction = prediction_by_id.get(qa_id)
        if prediction is None:
            missing_ids.append(qa_id)
            raw_prediction = ""
        else:
            raw_prediction = prediction.get("prediction_raw", prediction.get("prediction", ""))
        normalized_prediction = normalize_answer(raw_prediction)
        prediction_class = normalized_to_class.get(normalized_prediction)
        target_class = int(item["ans_cls"])
        scored_rows.append(
            {
                "qa_id": item["qa_id"],
                "video_id": item["video_id"],
                "type": item["type"],
                "sport": item["sport"],
                "question": item["question"],
                "answer": item["answer"],
                "ans_cls": target_class,
                "prediction_raw": raw_prediction,
                "prediction_normalized": normalized_prediction,
                "prediction_cls": prediction_class,
                "correct": prediction_class == target_class,
            }
        )

    if missing_ids and not args.allow_missing_predictions:
        raise SystemExit(
            f"Missing {len(missing_ids)} predictions; pass --allow-missing-predictions to score them as incorrect."
        )

    report = {
        "overall": calculate_metrics(scored_rows),
        "by_question_type": group_metrics(scored_rows, "type"),
        "by_sport": group_metrics(scored_rows, "sport"),
        "missing_prediction_count": len(missing_ids),
        "unknown_prediction_count": sum(
            row["prediction_cls"] is None for row in scored_rows
        ),
        "raw_prediction_variants": Counter(
            row["prediction_normalized"] for row in scored_rows if row["prediction_cls"] is None
        ).most_common(20),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    details_file = args.details_file or args.output_file.with_suffix(".details.jsonl")
    details_file.parent.mkdir(parents=True, exist_ok=True)
    with details_file.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
