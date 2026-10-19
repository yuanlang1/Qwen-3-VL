"""Command-line interface for the offline Sports-QA sampling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .dataset import SPLITS
from .pipeline import run_queries, run_scores, run_select, run_validate, summaries_json


DEFAULT_QUERY_BASE_URL = "https://llm-k8lrh2fh1trby9of.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata-root", type=Path, required=True, help="Directory containing train.json, val.json, and test.json")
    parser.add_argument("--video-root", type=Path, required=True, help="Root of extracted Sports-QA videos")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for resumable cache artifacts")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS), help="Splits to process (default: all)")
    parser.add_argument("--limit", type=int, help="Only use the first N QA records per split")
    parser.add_argument("--overwrite", action="store_true", help="Replace the selected stage's artifact when its configuration differs")


def _query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-model", default="qwen3-vl-flash")
    parser.add_argument("--query-base-url", default=DEFAULT_QUERY_BASE_URL)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--preview-resolution", type=int, default=224)
    parser.add_argument("--max-retries", type=int, default=3)


def _score_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--short-video-seconds", type=float, default=30.0)
    parser.add_argument("--medium-video-seconds", type=float, default=120.0)


def _selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--num-queries", type=int, choices=range(3, 8), default=7)


def _validation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--check-videos", action="store_true", help="Re-read selected source frames and verify stored decoder timestamps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, option_groups in {
        "queries": (_query_options,),
        "scores": (_score_options,),
        "select": (_selection_options,),
        "validate": (_selection_options, _validation_options),
        "all": (_query_options, _score_options, _selection_options, _validation_options),
    }.items():
        command = commands.add_parser(name)
        _common(command)
        for option_group in option_groups:
            option_group(command)
    return parser


def _run_all(args: Any):
    return run_queries(args) + run_scores(args) + run_select(args) + run_validate(args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    runners: dict[str, Callable[[Any], list[Any]]] = {
        "queries": run_queries,
        "scores": run_scores,
        "select": run_select,
        "validate": run_validate,
        "all": _run_all,
    }
    try:
        summaries = runners[args.command](args)
    except Exception as error:
        raise SystemExit(f"sportsqa-offline-sampling: {error}") from error
    print(summaries_json(summaries))
    if any(summary.failures for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
