"""Summarize Yang-style semantic judgments for Sports-QA."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def index_latest_judgments(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row in rows:
        qa_id = str(row["qa_id"])
        previous = indexed.get(qa_id)
        if previous is not None and previous.get("prediction_sha256") != row.get("prediction_sha256"):
            raise ValueError(f"Conflicting judgments for qa_id={qa_id}")
        indexed[qa_id] = row
    return indexed


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0, "correct": 0, "accuracy": None, "average_score": None}
    correct = sum(bool(row["semantic_match"]) for row in rows)
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "average_score": sum(float(row["semantic_score"]) for row in rows) / len(rows),
    }


def by_question_type(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["type"])].append(row)
    return {name: metrics(group) for name, group in sorted(groups.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Summarize only the first N manifest rows.")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    if not isinstance(manifest, list):
        raise ValueError("--manifest must contain a JSON list")
    if args.limit is not None:
        manifest = manifest[: args.limit]
    judgments = index_latest_judgments(load_jsonl(args.judgments))

    successful_rows = []
    missing_ids = []
    failed_ids = []
    judge_models = set()
    prompt_versions = set()
    for item in manifest:
        qa_id = str(item["qa_id"])
        row = judgments.get(qa_id)
        if row is None:
            missing_ids.append(qa_id)
            continue
        if row.get("status") != "ok":
            failed_ids.append(qa_id)
            continue
        if "semantic_match" not in row or "semantic_score" not in row:
            raise ValueError(f"Successful judgment is incomplete for qa_id={qa_id}")
        if str(row.get("type")) != str(item["type"]):
            raise ValueError(f"Question type mismatch for qa_id={qa_id}")
        successful_rows.append(row)
        judge_models.add(str(row.get("judge_model")))
        prompt_versions.add(str(row.get("judge_prompt_version")))

    incomplete_count = len(missing_ids) + len(failed_ids)
    if args.require_complete and incomplete_count:
        raise SystemExit(
            f"Semantic judgments are incomplete: {len(missing_ids)} missing, {len(failed_ids)} failed"
        )

    report = {
        "protocol": {
            "name": "Yang-style semantic judge with a substitute LLM judge",
            "judge_models": sorted(judge_models),
            "judge_prompt_versions": sorted(prompt_versions),
        },
        "expected_count": len(manifest),
        "judged_count": len(successful_rows),
        "coverage": len(successful_rows) / len(manifest) if manifest else None,
        "missing_judgment_count": len(missing_ids),
        "failed_judgment_count": len(failed_ids),
        "overall": metrics(successful_rows),
        "by_question_type": by_question_type(successful_rows),
        "missing_qa_ids": missing_ids[:20],
        "failed_qa_ids": failed_ids[:20],
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    final_result = {
        key: report["overall"][key] for key in ("count", "correct", "accuracy")
    }
    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
