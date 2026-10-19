"""The four resumable offline Sports-QA frame-selection stages."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import PIPELINE_VERSION
from .artifacts import FailureRecorder, StageArtifact, append_jsonl, fingerprint, index_records, prepare_jsonl_for_append
from .dataset import SportsQARecord, VideoIndex, file_sha256, load_split
from .query_bank import QUERY_COUNT, QueryBankGenerator
from .selection import adaptive_candidate_fps, select_slow_fast_frames
from .vision import FeatureCache, OpenCLIPFrameScorer, build_candidates, decode_preview_frames, inspect_video, read_frame_timestamps


@dataclass
class StageSummary:
    stage: str
    split: str
    expected: int
    processed: int = 0
    skipped: int = 0
    failures: int = 0


def _paths(output_dir: Path, split: str, num_queries: int = 7, frame_budget: int = 16) -> dict[str, Path]:
    return {
        "queries": output_dir / "queries" / f"{split}.jsonl",
        "scores": output_dir / "scores" / f"{split}.jsonl",
        "selection": output_dir / "selections" / split / f"q{num_queries}_f{frame_budget}.jsonl",
        "validation": output_dir / "reports" / f"validation_{split}_q{num_queries}_f{frame_budget}.json",
    }


def _config_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".config.json")


def _upstream_config_fingerprint(path: Path) -> str | None:
    config_path = _config_path(path)
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get("fingerprint")
    if not isinstance(value, str):
        raise ValueError(f"Invalid upstream configuration record: {config_path}")
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _records(metadata_root: Path, split: str, limit: int | None) -> list[SportsQARecord]:
    return load_split(metadata_root, split, limit)


def _stage_artifact(path: Path, config: dict[str, Any], overwrite: bool) -> tuple[StageArtifact, dict[str, dict[str, Any]]]:
    artifact = StageArtifact(path, _config_path(path), config)
    artifact.prepare(overwrite)
    return artifact, index_records(prepare_jsonl_for_append(path), path)


def _query_config(args: Any, split: str) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "queries",
        "split": split,
        "metadata_sha256": file_sha256(args.metadata_root / f"{split}.json"),
        "limit": args.limit,
        "query_generator": {
            "model": args.query_model,
            "base_url": args.query_base_url,
            "preview_count": args.preview_count,
            "preview_resolution": args.preview_resolution,
            "max_retries": args.max_retries,
        },
    }


def run_queries(args: Any) -> list[StageSummary]:
    index = VideoIndex(args.video_root)
    summaries: list[StageSummary] = []
    generator: QueryBankGenerator | None = None
    for split in args.splits:
        records = _records(args.metadata_root, split, args.limit)
        config = _query_config(args, split)
        paths = _paths(args.output_dir, split)
        _, complete = _stage_artifact(paths["queries"], config, args.overwrite)
        failure_log = FailureRecorder(args.output_dir / "failures" / f"queries_{split}.jsonl", fingerprint(config))
        summary = StageSummary("queries", split, len(records))
        pending = [record for record in records if record.key not in complete]
        summary.skipped = len(records) - len(pending)
        by_video: dict[str, list[SportsQARecord]] = defaultdict(list)
        for record in pending:
            by_video[record.video].append(record)
        with paths["queries"].open("a", encoding="utf-8") as output:
            for video_id, video_records in by_video.items():
                try:
                    video_path = index.resolve(video_id)
                    previews, preview_indices = decode_preview_frames(
                        video_path, args.preview_count, args.preview_resolution
                    )
                except Exception as error:
                    for record in video_records:
                        failure_log.record(split=split, qa_id=record.qa_id, video=record.video, error=error)
                        summary.failures += 1
                    continue
                if generator is None:
                    generator = QueryBankGenerator(
                        model=args.query_model,
                        base_url=args.query_base_url,
                        preview_count=args.preview_count,
                        preview_resolution=args.preview_resolution,
                        max_retries=args.max_retries,
                    )
                for record in video_records:
                    try:
                        generated = generator.generate_from_previews(record.question, previews, preview_indices)
                        append_jsonl(output, {
                            "split": split,
                            "qa_id": record.qa_id,
                            "video": record.video,
                            "resolved_video": index.relative_path(video_path),
                            "question": record.question,
                            **generated,
                            "query_generator": args.query_model,
                        })
                        summary.processed += 1
                    except Exception as error:
                        failure_log.record(split=split, qa_id=record.qa_id, video=record.video, error=error)
                        summary.failures += 1
        summaries.append(summary)
    return summaries


def _score_config(args: Any, split: str, query_path: Path) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "scores",
        "split": split,
        "query_config_fingerprint": _upstream_config_fingerprint(query_path),
        "clip": {
            "model": args.clip_model,
            "pretrained": args.clip_pretrained,
            "short_video_seconds": args.short_video_seconds,
            "medium_video_seconds": args.medium_video_seconds,
        },
    }


def _valid_query_record(record: dict[str, Any], path: Path) -> list[str]:
    queries = record.get("query_bank")
    if not isinstance(queries, list) or len(queries) != QUERY_COUNT or not all(isinstance(query, str) and query.strip() for query in queries):
        raise ValueError(f"{path} has an invalid seven-query bank for {record.get('split')}:{record.get('qa_id')}")
    return queries


def run_scores(args: Any) -> list[StageSummary]:
    index = VideoIndex(args.video_root)
    summaries: list[StageSummary] = []
    scorer: OpenCLIPFrameScorer | None = None
    feature_config = {
        "pipeline_version": PIPELINE_VERSION,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "short_video_seconds": args.short_video_seconds,
        "medium_video_seconds": args.medium_video_seconds,
    }
    cache = FeatureCache(args.output_dir / "cache" / "features", feature_config)
    for split in args.splits:
        paths = _paths(args.output_dir, split)
        query_records = prepare_jsonl_for_append(paths["queries"])
        query_by_key = index_records(query_records, paths["queries"])
        config = _score_config(args, split, paths["queries"])
        _, complete = _stage_artifact(paths["scores"], config, args.overwrite)
        failure_log = FailureRecorder(args.output_dir / "failures" / f"scores_{split}.jsonl", fingerprint(config))
        summary = StageSummary("scores", split, len(query_records))
        pending = [record for key, record in query_by_key.items() if key not in complete]
        summary.skipped = len(query_records) - len(pending)
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pending:
            by_video[str(record.get("video", ""))].append(record)
        with paths["scores"].open("a", encoding="utf-8") as output:
            for video_id, video_records in by_video.items():
                try:
                    queries_by_record = [_valid_query_record(record, paths["queries"]) for record in video_records]
                    video_path = index.resolve(video_id)
                    info = inspect_video(video_path)
                    candidate_fps = adaptive_candidate_fps(
                        info.duration_seconds, args.short_video_seconds, args.medium_video_seconds
                    )
                    candidate_indices = build_candidates(info, candidate_fps)
                    candidate_timestamps = read_frame_timestamps(video_path, candidate_indices)
                    cache_metadata = {
                        "video_id": video_id,
                        "resolved_video": index.relative_path(video_path),
                        "total_frames": info.total_frames,
                        "average_fps": info.average_fps,
                        "candidate_fps": candidate_fps,
                        "candidate_frame_indices": candidate_indices,
                    }
                    features = cache.load(video_id, video_path, cache_metadata)
                    if features is None:
                        if scorer is None:
                            scorer = OpenCLIPFrameScorer(
                                model_name=args.clip_model,
                                pretrained=args.clip_pretrained,
                                device=args.device,
                                image_batch_size=args.image_batch_size,
                                text_batch_size=args.text_batch_size,
                            )
                        features = scorer.encode_video_frames(video_path, candidate_indices).numpy()
                        cache.save(video_id, video_path, cache_metadata, features)
                    if scorer is None:
                        scorer = OpenCLIPFrameScorer(
                            model_name=args.clip_model,
                            pretrained=args.clip_pretrained,
                            device=args.device,
                            image_batch_size=args.image_batch_size,
                            text_batch_size=args.text_batch_size,
                        )
                    all_queries = [query for queries in queries_by_record for query in queries]
                    score_matrix = scorer.score_queries(all_queries, scorer.torch.from_numpy(features)).numpy()
                except Exception as error:
                    for record in video_records:
                        failure_log.record(split=split, qa_id=str(record.get("qa_id", "?")), video=video_id, error=error)
                        summary.failures += 1
                    continue
                for position, record in enumerate(video_records):
                    start = position * QUERY_COUNT
                    end = start + QUERY_COUNT
                    append_jsonl(output, {
                        "split": split,
                        "qa_id": str(record["qa_id"]),
                        "video": video_id,
                        "resolved_video": index.relative_path(video_path),
                        "question": record["question"],
                        "query_bank": queries_by_record[position],
                        "source_video": {
                            "total_frames": info.total_frames,
                            "average_fps": info.average_fps,
                        },
                        "candidate_frames": {
                            "fps": candidate_fps,
                            "frame_indices": candidate_indices,
                            "timestamps": candidate_timestamps,
                        },
                        "query_frame_scores": score_matrix[start:end].tolist(),
                        "frame_scorer": {
                            "model": args.clip_model,
                            "pretrained": args.clip_pretrained,
                            "score_type": "cosine_similarity",
                        },
                    })
                    summary.processed += 1
        summaries.append(summary)
    return summaries


def _selection_config(args: Any, split: str, score_path: Path) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "select",
        "split": split,
        "metadata_sha256": file_sha256(args.metadata_root / f"{split}.json"),
        "score_config_fingerprint": _upstream_config_fingerprint(score_path),
        "limit": args.limit,
        "num_queries": args.num_queries,
        "frame_budget": args.frame_budget,
        "alpha": 0.5,
        "gaussian_radius": 4,
        "gaussian_sigma": 1.0,
        "fast_ratio": 0.25,
    }


def run_select(args: Any) -> list[StageSummary]:
    summaries: list[StageSummary] = []
    for split in args.splits:
        records = _records(args.metadata_root, split, args.limit)
        paths = _paths(args.output_dir, split, args.num_queries, args.frame_budget)
        default_paths = _paths(args.output_dir, split)
        score_records = prepare_jsonl_for_append(default_paths["scores"])
        score_by_key = index_records(score_records, default_paths["scores"])
        config = _selection_config(args, split, default_paths["scores"])
        _, complete = _stage_artifact(paths["selection"], config, args.overwrite)
        failure_log = FailureRecorder(args.output_dir / "failures" / f"select_{split}_q{args.num_queries}_f{args.frame_budget}.jsonl", fingerprint(config))
        summary = StageSummary("select", split, len(records))
        with paths["selection"].open("a", encoding="utf-8") as output:
            for record in records:
                if record.key in complete:
                    summary.skipped += 1
                    continue
                try:
                    score = score_by_key[record.key]
                    if score.get("video") != record.video or score.get("question") != record.question:
                        raise ValueError(f"Score record identity does not match metadata for {record.key}")
                    queries = _valid_query_record(score, default_paths["scores"])
                    if len(queries) < args.num_queries:
                        raise ValueError(f"Score record has fewer than {args.num_queries} queries")
                    candidates = score["candidate_frames"]
                    candidate_indices = [int(value) for value in candidates["frame_indices"]]
                    candidate_timestamps = [float(value) for value in candidates["timestamps"]]
                    if len(candidate_indices) != len(candidate_timestamps):
                        raise ValueError("Candidate indices and decoder timestamps differ in length")
                    selected = select_slow_fast_frames(
                        score["query_frame_scores"][: args.num_queries],
                        candidate_indices,
                        budget=args.frame_budget,
                    )
                    timestamp_by_index = dict(zip(candidate_indices, candidate_timestamps))
                    selected_timestamps = [timestamp_by_index[index] for index in selected.frame_indices]
                    append_jsonl(output, {
                        "split": split,
                        "qa_id": record.qa_id,
                        "video": record.video,
                        "resolved_video": score["resolved_video"],
                        "question": record.question,
                        "source_video": score["source_video"],
                        "selection": {
                            "num_queries": args.num_queries,
                            "frame_budget": args.frame_budget,
                            "alpha": 0.5,
                            "gaussian_radius": 4,
                            "gaussian_sigma": 1.0,
                            "fast_ratio": 0.25,
                        },
                        "selected_frame_indices": selected.frame_indices,
                        "selected_timestamps": selected_timestamps,
                        "sampling_types": selected.sampling_types,
                        "clip_ids": selected.clip_ids,
                        "relevance_scores": selected.relevance_scores,
                        "actual_frame_count": len(selected.frame_indices),
                        "effective_alpha": selected.effective_alpha,
                        "clips": [list(clip) for clip in selected.clips],
                    })
                    summary.processed += 1
                except Exception as error:
                    failure_log.record(split=split, qa_id=record.qa_id, video=record.video, error=error)
                    summary.failures += 1
        summaries.append(summary)
    return summaries


def _validate_record(record: dict[str, Any], expected: SportsQARecord, args: Any) -> str | None:
    if record.get("video") != expected.video or record.get("question") != expected.question:
        return "video or question does not match source metadata"
    selection = record.get("selection", {})
    if selection.get("num_queries") != args.num_queries or selection.get("frame_budget") != args.frame_budget:
        return "selection configuration does not match the requested output"
    frame_indices = record.get("selected_frame_indices", [])
    timestamps = record.get("selected_timestamps", [])
    kinds = record.get("sampling_types", [])
    clip_ids = record.get("clip_ids", [])
    scores = record.get("relevance_scores", [])
    if not all(isinstance(values, list) for values in (frame_indices, timestamps, kinds, clip_ids, scores)):
        return "selection arrays are missing"
    if len(frame_indices) != len(timestamps) or len(frame_indices) != len(kinds) or len(frame_indices) != len(clip_ids) or len(frame_indices) != len(scores):
        return "selection arrays differ in length"
    if record.get("actual_frame_count") != len(frame_indices):
        return "actual_frame_count differs from selected frame count"
    total_frames = record.get("source_video", {}).get("total_frames")
    if not isinstance(total_frames, int) or total_frames <= 0:
        return "source video frame count is invalid"
    if any(not isinstance(index, int) or index < 0 or index >= total_frames for index in frame_indices):
        return "selected frame index is out of source-video bounds"
    if frame_indices != sorted(frame_indices) or len(set(frame_indices)) != len(frame_indices):
        return "selected frame indices are not sorted and unique"
    if any(float(later) < float(earlier) for earlier, later in zip(timestamps, timestamps[1:])):
        return "selected decoder timestamps are not time ordered"
    return None


def run_validate(args: Any) -> list[StageSummary]:
    summaries: list[StageSummary] = []
    index = VideoIndex(args.video_root) if args.check_videos else None
    for split in args.splits:
        records = _records(args.metadata_root, split, args.limit)
        paths = _paths(args.output_dir, split, args.num_queries, args.frame_budget)
        selection_path = paths["selection"]
        config = {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "validate",
            "split": split,
            "metadata_sha256": file_sha256(args.metadata_root / f"{split}.json"),
            "selection_config_fingerprint": _upstream_config_fingerprint(selection_path),
            "limit": args.limit,
            "num_queries": args.num_queries,
            "frame_budget": args.frame_budget,
            "check_videos": bool(args.check_videos),
        }
        report_artifact = StageArtifact(paths["validation"], _config_path(paths["validation"]), config)
        report_artifact.prepare(args.overwrite)
        selected_records = prepare_jsonl_for_append(selection_path)
        selected_by_key = index_records(selected_records, selection_path)
        expected_by_key = {record.key: record for record in records}
        invalid: list[dict[str, str]] = []
        missing = [key for key in expected_by_key if key not in selected_by_key]
        extras = [key for key in selected_by_key if key not in expected_by_key]
        for key, expected in expected_by_key.items():
            selected = selected_by_key.get(key)
            if selected is None:
                continue
            error = _validate_record(selected, expected, args)
            if error:
                invalid.append({"key": key, "error": error})
        video_errors: list[dict[str, str]] = []
        if index is not None:
            by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key in expected_by_key:
                if key in selected_by_key:
                    by_video[str(selected_by_key[key]["video"])].append(selected_by_key[key])
            for video_id, video_selections in by_video.items():
                try:
                    video_path = index.resolve(video_id)
                    actual_info = inspect_video(video_path)
                    first = video_selections[0]["source_video"]
                    if actual_info.total_frames != first["total_frames"]:
                        raise ValueError("source frame count no longer matches the selection artifact")
                    indices = sorted({index for selection in video_selections for index in selection["selected_frame_indices"]})
                    actual_timestamps = dict(zip(indices, read_frame_timestamps(video_path, indices)))
                    for selection in video_selections:
                        for frame_index, timestamp in zip(selection["selected_frame_indices"], selection["selected_timestamps"]):
                            if abs(actual_timestamps[frame_index] - timestamp) > 0.001:
                                raise ValueError("selected timestamp no longer matches decoder output")
                except Exception as error:
                    video_errors.append({"video": video_id, "error": str(error)})
        failures = len(missing) + len(extras) + len(invalid) + len(video_errors)
        report = {
            "split": split,
            "selection_path": str(selection_path),
            "expected_count": len(records),
            "selection_count": len(selected_records),
            "coverage": len(records) - len(missing),
            "coverage_ratio": (len(records) - len(missing)) / len(records) if records else 1.0,
            "missing_count": len(missing),
            "missing_examples": missing[:100],
            "extra_count": len(extras),
            "extra_examples": extras[:100],
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:100],
            "video_check_enabled": bool(index),
            "video_error_count": len(video_errors),
            "video_error_examples": video_errors[:100],
            "valid": failures == 0,
        }
        _write_json(paths["validation"], report)
        summaries.append(StageSummary("validate", split, len(records), processed=len(records) - failures, failures=failures))
    return summaries


def summaries_json(summaries: Iterable[StageSummary]) -> str:
    return json.dumps([asdict(summary) for summary in summaries], ensure_ascii=False, indent=2)
