"""Independent command entry points; no imports from the existing Qwen route."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from .adapters import CONVERTERS
from .pipeline import run_normalize, run_queries, run_scores, run_select, run_validate, selection_path


def _video_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-root", type=Path, default=Path("."), help="Relative video paths resolve under this root (default: working directory)")
    parser.add_argument("--video-index", type=Path, help="Optional JSON map from video IDs to exact paths")


def _query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-model", default="qwen3-vl-flash")
    parser.add_argument("--query-base-url", default="https://llm-k8lrh2fh1trby9of.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--preview-resolution", type=int, default=224)
    parser.add_argument("--max-retries", type=int, default=3)


def _score_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", type=Path, help="Query JSONL; defaults to OUTPUT_DIR/queries.jsonl")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--short-video-seconds", type=float, default=30.0)
    parser.add_argument("--medium-video-seconds", type=float, default=120.0)


def _selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scores", type=Path, help="Score JSONL; defaults to OUTPUT_DIR/scores.jsonl")
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--num-queries", type=int, choices=range(3, 8), default=7)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gaussian-radius", type=int, default=4)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--fast-ratio", type=float, default=0.25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sports-aware query banks and offline Slow-Fast sampling across datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize", help="Convert a dataset annotation into standard sports QA JSONL")
    normalize.add_argument("--format", choices=sorted(CONVERTERS), required=True)
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--overwrite", action="store_true")
    for name in ("queries", "scores", "select", "validate", "all"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--overwrite", action="store_true", help="Replace stage artifacts; prefer a new directory when changing upstream settings")
        if name in {"queries", "validate", "all"}:
            command.add_argument("--input", type=Path, required=name != "validate", help="Standard input; optional coverage check for validate")
            command.add_argument("--limit", type=int, help="Limit input records for a smoke test; use a dedicated output directory")
        if name in {"queries", "scores", "validate", "all"}:
            _video_options(command)
        if name in {"queries", "all"}:
            _query_options(command)
        if name in {"scores", "all"}:
            _score_options(command)
        if name in {"select", "validate", "all"}:
            _selection_options(command)
        if name == "validate":
            command.add_argument("--selections", type=Path, help="Selection JSONL; otherwise derive from score/config fingerprint")
        if name in {"validate", "all"}:
            command.add_argument("--check-videos", action="store_true", help="Decode selected original frames and verify their timestamps")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("limit", "preview_count", "preview_resolution", "max_retries", "image_batch_size", "text_batch_size", "frame_budget"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "short_video_seconds"):
        if not 0 < args.short_video_seconds < args.medium_video_seconds or not math.isfinite(args.medium_video_seconds):
            raise ValueError("Require 0 < --short-video-seconds < --medium-video-seconds, both finite")
    if hasattr(args, "alpha"):
        if args.alpha < 0 or not math.isfinite(args.alpha):
            raise ValueError("--alpha must be finite and nonnegative")
        if args.gaussian_radius < 0 or args.gaussian_sigma <= 0 or not math.isfinite(args.gaussian_sigma):
            raise ValueError("Require nonnegative --gaussian-radius and positive finite --gaussian-sigma")
        if not 0 <= args.fast_ratio <= 1:
            raise ValueError("--fast-ratio must be in [0, 1]")
    if args.command == "all" and (args.queries is not None or args.scores is not None):
        raise ValueError("all uses its own upstream artifacts; use individual stages for external queries/scores")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summaries = []
    try:
        validate_args(args)
        runners = {"normalize": run_normalize, "queries": run_queries, "scores": run_scores, "select": run_select, "validate": run_validate}
        stages = ("queries", "scores", "select", "validate") if args.command == "all" else (args.command,)
        for stage in stages:
            if stage == "validate" and args.command == "all":
                args.selections = selection_path(args)
            summary = runners[stage](args)
            summaries.append(asdict(summary))
            if stage == "select":
                summaries[-1]["output"] = str(selection_path(args))
            if summary.failures:
                break
    except Exception as error:
        raise SystemExit(f"sports-video-sampling: {error}") from error
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if any(summary["failures"] for summary in summaries):
        raise SystemExit(1)
