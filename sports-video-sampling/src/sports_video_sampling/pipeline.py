"""Resumable sports sampling stages independent of dataset-specific metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import PIPELINE_VERSION
from .adapters import CONVERTERS
from .artifacts import FailureRecorder, StageArtifact, append_jsonl, fingerprint, index_records, prepare_jsonl_for_append, read_jsonl
from .consumer import decode_selected_frames, validate_selection
from .query_bank import QueryBankGenerator, select_query_prefix
from .records import VideoResolver, file_sha256, load_records, parse_record, read_rows
from .scorer import FeatureCache, OpenCLIPFrameScorer
from .selection import adaptive_candidate_fps, candidate_frame_indices, select_slow_fast_frames
from .video_io import decode_preview_frames, inspect_video, read_frame_timestamps


@dataclass
class StageSummary:
    stage: str
    expected: int
    processed: int = 0
    skipped: int = 0
    failures: int = 0


def _resolver(args: Any) -> VideoResolver:
    return VideoResolver(args.video_root, args.video_index)


def _artifact_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required upstream artifact does not exist: {path}")
    rows, _ = read_jsonl(path, recover_truncated_tail=False)
    index_records(rows, path)
    for row in rows:
        row["id"] = str(row["id"])
    return rows


def _prepare(path: Path, config: dict[str, Any], overwrite: bool) -> dict[str, dict[str, Any]]:
    StageArtifact(path, path.with_suffix(".config.json"), config).prepare(overwrite)
    return index_records(prepare_jsonl_for_append(path), path)


def _source_states(videos: list[str], resolver: VideoResolver) -> list[dict[str, Any]]:
    states = []
    for video in sorted(set(videos)):
        try:
            path = resolver.resolve(video)
            stat = path.stat()
            states.append({"video": video, "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except (FileNotFoundError, ValueError):
            states.append({"video": video, "unresolved": True})
    return states


def _video_config(args: Any, videos: list[str], resolver: VideoResolver) -> dict[str, Any]:
    return {
        "video_root": str(args.video_root.resolve()),
        "video_index_sha256": file_sha256(args.video_index) if args.video_index else None,
        "source_videos": _source_states(videos, resolver),
        "decoder": "decord",
    }


def _failure_log(args: Any, stage: str, config: dict[str, Any]) -> FailureRecorder:
    return FailureRecorder(args.output_dir / "failures" / f"{stage}.jsonl", fingerprint(config))


def _group_videos(rows: list[dict[str, Any]], resolver: VideoResolver, failures: FailureRecorder, summary: StageSummary):
    groups: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            groups[resolver.resolve(row["video"])].append(row)
        except Exception as error:
            failures.record(sample_id=row["id"], video=row["video"], error=error)
            summary.failures += 1
    return groups


def run_normalize(args: Any) -> StageSummary:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Normalization output must differ from the source input")
    if args.output.suffix.lower() != ".jsonl":
        raise ValueError("Normalization output must use the .jsonl extension")
    converted = [parse_record(CONVERTERS[args.format](row)).to_dict() for row in read_rows(args.input)]
    index_records(converted, args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; use --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        for row in converted:
            append_jsonl(output, row)
    temporary.replace(args.output)
    return StageSummary("normalize", len(converted), processed=len(converted))


def run_queries(args: Any) -> StageSummary:
    records = [record.to_dict() for record in load_records(args.input, args.limit)]
    resolver = _resolver(args)
    config = {
        "pipeline_version": PIPELINE_VERSION, "stage": "queries",
        "input_sha256": file_sha256(args.input), "limit": args.limit,
        **_video_config(args, [row["video"] for row in records], resolver),
        "model": args.query_model, "base_url": args.query_base_url,
        "preview_count": args.preview_count, "preview_resolution": args.preview_resolution,
        "max_retries": args.max_retries,
        "prompt_sha256": fingerprint(QueryBankGenerator.build_prompt("{question}")),
    }
    path = args.output_dir / "queries.jsonl"
    if path.resolve() == args.input.resolve():
        raise ValueError("Query output must differ from the standard input")
    complete = _prepare(path, config, args.overwrite)
    summary = StageSummary("queries", len(records), skipped=sum(row["id"] in complete for row in records))
    failures = _failure_log(args, "queries", config)
    pending = [row for row in records if row["id"] not in complete]
    generator = None
    with path.open("a", encoding="utf-8") as output:
        for video_path, rows in _group_videos(pending, resolver, failures, summary).items():
            try:
                previews, indices = decode_preview_frames(video_path, args.preview_count, args.preview_resolution)
                if generator is None:
                    generator = QueryBankGenerator(
                        model=args.query_model, base_url=args.query_base_url,
                        preview_count=args.preview_count, preview_resolution=args.preview_resolution,
                        max_retries=args.max_retries,
                    )
            except Exception as error:
                for row in rows:
                    failures.record(sample_id=row["id"], video=row["video"], error=error)
                    summary.failures += 1
                continue
            for row in rows:
                try:
                    generated = generator.generate_from_previews(row["question"], previews, indices)
                    append_jsonl(output, {
                        **row, **generated, "resolved_video": str(video_path),
                        "query_generator": args.query_model, "sport_type_source": "preview_inference",
                    })
                    summary.processed += 1
                except Exception as error:
                    failures.record(sample_id=row["id"], video=row["video"], error=error)
                    summary.failures += 1
    return summary


def _score_path(args: Any) -> Path:
    return args.scores if args.scores is not None else args.output_dir / "scores.jsonl"


def run_scores(args: Any) -> StageSummary:
    query_path = args.queries if args.queries is not None else args.output_dir / "queries.jsonl"
    rows = _artifact_rows(query_path)
    resolver = _resolver(args)
    feature_config = {
        "pipeline_version": PIPELINE_VERSION, "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained, "decoder": "decord",
        "short_video_seconds": args.short_video_seconds, "medium_video_seconds": args.medium_video_seconds,
    }
    config = {
        **feature_config, "stage": "scores", "queries_sha256": file_sha256(query_path),
        **_video_config(args, [row["video"] for row in rows], resolver),
        "device": args.device,
    }
    path = args.output_dir / "scores.jsonl"
    if path.resolve() == query_path.resolve():
        raise ValueError("Score output must differ from its query input")
    complete = _prepare(path, config, args.overwrite)
    failures = _failure_log(args, "scores", config)
    summary = StageSummary("scores", len(rows), skipped=sum(row["id"] in complete for row in rows))
    pending = []
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            if row["id"] in complete:
                continue
            try:
                parse_record(row)
                bank = QueryBankGenerator.parse_response(json.dumps(row))
                pending.append({**row, **bank})
            except Exception as error:
                failures.record(sample_id=row["id"], video=str(row.get("video", "")), error=error)
                summary.failures += 1
        scorer = None
        cache = FeatureCache(args.output_dir / "cache" / "features", feature_config)
        for video_path, video_rows in _group_videos(pending, resolver, failures, summary).items():
            try:
                info = inspect_video(video_path)
                fps = adaptive_candidate_fps(info.duration_seconds, args.short_video_seconds, args.medium_video_seconds)
                candidates = candidate_frame_indices(info.total_frames, info.average_fps, fps)
                timestamps = read_frame_timestamps(video_path, candidates)
                cache_metadata = {
                    "source_video": asdict(info), "candidate_fps": fps,
                    "candidate_frame_indices": candidates, "candidate_timestamps": timestamps,
                }
                features = cache.load(str(video_path), video_path, cache_metadata)
                if scorer is None:
                    scorer = OpenCLIPFrameScorer(
                        model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.device,
                        image_batch_size=args.image_batch_size, text_batch_size=args.text_batch_size,
                    )
                if features is None:
                    features = scorer.encode_video_frames(video_path, candidates).numpy()
                    cache.save(str(video_path), video_path, cache_metadata, features)
                queries = [query for row in video_rows for query in row["query_bank"]]
                matrix = scorer.score_queries(queries, scorer.torch.from_numpy(features)).numpy()
                if matrix.shape != (7 * len(video_rows), len(candidates)) or not np.isfinite(matrix).all():
                    raise ValueError("CLIP returned invalid query/frame scores")
            except Exception as error:
                for row in video_rows:
                    failures.record(sample_id=row["id"], video=row["video"], error=error)
                    summary.failures += 1
                continue
            for position, row in enumerate(video_rows):
                append_jsonl(output, {
                    **row, "resolved_video": str(video_path), "decoder": "decord",
                    "source_video": asdict(info),
                    "candidate_frames": {"fps": fps, "frame_indices": candidates, "timestamps": timestamps},
                    "query_frame_scores": matrix[position * 7:(position + 1) * 7].tolist(),
                    "frame_scorer": {"model": args.clip_model, "pretrained": args.clip_pretrained, "score_type": "cosine_similarity"},
                })
                summary.processed += 1
    return summary


def _selection_config(args: Any) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION, "stage": "select",
        "scores_sha256": file_sha256(_score_path(args)),
        "num_queries": args.num_queries, "frame_budget": args.frame_budget,
        "alpha": args.alpha, "gaussian_radius": args.gaussian_radius,
        "gaussian_sigma": args.gaussian_sigma, "fast_ratio": args.fast_ratio,
    }


def selection_path(args: Any) -> Path:
    config = _selection_config(args)
    return args.output_dir / "selections" / f"q{args.num_queries}_f{args.frame_budget}_{fingerprint(config)[:12]}.jsonl"


def run_select(args: Any) -> StageSummary:
    score_path = _score_path(args)
    rows = _artifact_rows(score_path)
    config, path = _selection_config(args), selection_path(args)
    if path.resolve() == score_path.resolve():
        raise ValueError("Selection output must differ from its score input")
    complete = _prepare(path, config, args.overwrite)
    failures = _failure_log(args, "select", config)
    summary = StageSummary("select", len(rows), skipped=sum(row["id"] in complete for row in rows))
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            if row["id"] in complete:
                continue
            try:
                record = parse_record(row)
                bank = QueryBankGenerator.parse_response(json.dumps(row))
                queries = select_query_prefix(bank["query_bank"], args.num_queries)
                candidates = row["candidate_frames"]
                indices, timestamps = candidates["frame_indices"], candidates["timestamps"]
                matrix = np.asarray(row["query_frame_scores"], dtype=np.float32)
                if matrix.shape != (7, len(indices)) or len(indices) != len(timestamps):
                    raise ValueError("Candidate indices, timestamps, and seven-query score matrix do not align")
                if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                    raise ValueError("Candidate frame indices must be integers")
                if any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in timestamps):
                    raise ValueError("Candidate timestamps must be numbers")
                if not np.isfinite(timestamps).all() or any(value < 0 for value in timestamps) or any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
                    raise ValueError("Candidate timestamps must be finite, nonnegative, and chronological")
                total_frames = row["source_video"]["total_frames"]
                if not isinstance(total_frames, int) or total_frames <= 0 or any(index >= total_frames for index in indices):
                    raise ValueError("Candidate indices exceed the source-video bounds")
                selected = select_slow_fast_frames(
                    matrix[:args.num_queries], indices, budget=args.frame_budget, alpha=args.alpha,
                    gaussian_radius=args.gaussian_radius, gaussian_sigma=args.gaussian_sigma, fast_ratio=args.fast_ratio,
                )
                timestamp_by_index = dict(zip(indices, timestamps))
                result = {
                    **record.to_dict(), **bank,
                    "resolved_video": row.get("resolved_video", row["video"]),
                    "decoder": row.get("decoder"), "source_video": row["source_video"],
                    "selection": {key: config[key] for key in ("num_queries", "frame_budget", "alpha", "gaussian_radius", "gaussian_sigma", "fast_ratio")},
                    "selected_queries": queries, "candidate_count": len(indices),
                    "selected_frame_indices": selected.frame_indices,
                    "selected_timestamps": [timestamp_by_index[index] for index in selected.frame_indices],
                    "sampling_types": selected.sampling_types, "clip_ids": selected.clip_ids,
                    "relevance_scores": selected.relevance_scores, "actual_frame_count": len(selected.frame_indices),
                    "effective_alpha": selected.effective_alpha, "clips": [list(clip) for clip in selected.clips],
                }
                validate_selection(result)
                append_jsonl(output, result)
                summary.processed += 1
            except Exception as error:
                failures.record(sample_id=row["id"], video=str(row.get("video", "")), error=error)
                summary.failures += 1
    return summary


def run_validate(args: Any) -> StageSummary:
    path = args.selections if args.selections is not None else selection_path(args)
    rows = _artifact_rows(path)
    by_id = index_records(rows, path)
    expected = {record.id: record for record in load_records(args.input, args.limit)} if args.input else None
    errors = []
    if expected is not None:
        errors.extend({"id": key, "error": "missing selection"} for key in expected if key not in by_id)
        errors.extend({"id": key, "error": "unexpected selection"} for key in by_id if key not in expected)
    resolver = _resolver(args) if args.check_videos else None
    for row in rows:
        try:
            validate_selection(row)
            if expected is not None and row["id"] in expected:
                source = expected[row["id"]]
                if row["video"] != source.video or row["question"] != source.question:
                    raise ValueError("Video or question differs from input")
            if resolver is not None:
                decode_selected_frames(resolver.resolve(row["video"]), row)
        except Exception as error:
            errors.append({"id": row["id"], "error": str(error)})
    report = {
        "selection_path": str(path.resolve()), "selection_count": len(rows),
        "expected_count": len(expected) if expected is not None else None,
        "coverage_checked": expected is not None, "video_check_enabled": args.check_videos,
        "failure_count": len(errors), "errors": errors, "valid": not errors,
    }
    report_path = args.output_dir / "reports" / f"validation_{path.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return StageSummary("validate", len(expected) if expected is not None else len(rows), processed=len(rows), failures=len(errors))
