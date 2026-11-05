import json
from types import SimpleNamespace

import numpy as np
import pytest

from sports_video_sampling import pipeline
from sports_video_sampling.artifacts import ConfigMismatchError
from sports_video_sampling.cli import build_parser, main
from sports_video_sampling.consumer import load_selections
from sports_video_sampling.query_bank import QueryBankGenerator
from sports_video_sampling.video_io import VideoInfo


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values


def setup_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    rows = [{"id": str(index), "video": "match.mp4", "question": question,
             "metadata": {"dataset": "another_sports_dataset", "split": "challenge", "answer": "SECRET"}}
            for index, question in enumerate(("early action", "late action"))]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "match.mp4").write_bytes(b"test-video")
    counts = {"previews": 0, "encode": 0, "generate": 0}

    def previews(*args):
        counts["previews"] += 1
        return ["preview"], [0]

    def generate(self, question, previews, indices):
        counts["generate"] += 1
        assert "SECRET" not in question
        return {"sport_type": "basketball", "sport_type_confidence": "high", "query_bank": [f"{question} evidence {index}" for index in range(7)], "preview_frame_indices": indices}

    class Scorer:
        torch = SimpleNamespace(from_numpy=Tensor)

        def __init__(self, **kwargs):
            pass

        def encode_video_frames(self, path, indices):
            counts["encode"] += 1
            return Tensor(np.zeros((len(indices), 2)))

        def score_queries(self, queries, features):
            matrix = np.zeros((len(queries), len(features.numpy())))
            for index, query in enumerate(queries):
                matrix[index, 5 if "early" in query else 25] = 1
            return Tensor(matrix)

    monkeypatch.setattr(pipeline, "decode_preview_frames", previews)
    monkeypatch.setattr(QueryBankGenerator, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(QueryBankGenerator, "generate_from_previews", generate)
    monkeypatch.setattr(pipeline, "OpenCLIPFrameScorer", Scorer)
    monkeypatch.setattr(pipeline, "inspect_video", lambda path: VideoInfo(80, 10.0))
    monkeypatch.setattr(pipeline, "read_frame_timestamps", lambda path, indices: [index / 10 for index in indices])
    base = ["--input", str(source), "--video-root", str(tmp_path), "--output-dir", str(tmp_path / "output"), "--frame-budget", "8"]
    return source, counts, base


def test_all_is_per_question_cached_resumable_and_independent_of_dataset(tmp_path, monkeypatch, capsys):
    source, counts, base = setup_pipeline(tmp_path, monkeypatch)
    main(["all", *base])
    assert counts == {"previews": 1, "encode": 1, "generate": 2}
    args = build_parser().parse_args(["all", *base])
    path = pipeline.selection_path(args)
    selections = load_selections(path)
    assert set(selections) == {"0", "1"}
    assert selections["0"]["selected_frame_indices"] != selections["1"]["selected_frame_indices"]
    assert selections["0"]["metadata"]["split"] == "challenge"
    main(["all", *base])
    assert counts == {"previews": 1, "encode": 1, "generate": 2}
    main(["scores", "--video-root", str(tmp_path), "--output-dir", str(tmp_path / "output"), "--overwrite"])
    assert counts["encode"] == 1


def test_sampling_parameters_create_separate_outputs_and_reuse_scores(tmp_path, monkeypatch):
    _, counts, base = setup_pipeline(tmp_path, monkeypatch)
    main(["all", *base])
    args = build_parser().parse_args(["select", "--output-dir", str(tmp_path / "output"), "--frame-budget", "8", "--fast-ratio", "0.5", "--alpha", "1", "--gaussian-radius", "0", "--gaussian-sigma", "2", "--num-queries", "3"])
    assert pipeline.run_select(args).failures == 0
    rows = load_selections(pipeline.selection_path(args))
    assert rows["0"]["sampling_types"].count("fast") == 4
    assert rows["0"]["selection"]["gaussian_sigma"] == 2
    assert rows["0"]["selection"]["gaussian_normalization"] == "continuous"
    config_path = pipeline.selection_path(args).with_suffix(".config.json")
    assert json.loads(config_path.read_text(encoding="utf-8"))["config"]["gaussian_normalization"] == "continuous"
    assert len(rows["0"]["selected_queries"]) == 3
    assert len(list((tmp_path / "output" / "selections").glob("*.jsonl"))) == 2
    assert counts["generate"] == 2 and counts["encode"] == 1


def test_changed_input_rejects_resume_before_api_calls(tmp_path, monkeypatch):
    source, counts, base = setup_pipeline(tmp_path, monkeypatch)
    main(["all", *base])
    source.write_text(source.read_text(encoding="utf-8").replace("early action", "other action"), encoding="utf-8")
    args = build_parser().parse_args(["all", *base])
    with pytest.raises(ConfigMismatchError):
        pipeline.run_queries(args)
    assert counts["generate"] == 2


def test_all_stops_on_failures_and_can_retry_api_failures(tmp_path, monkeypatch):
    _, counts, base = setup_pipeline(tmp_path, monkeypatch)
    generate = QueryBankGenerator.generate_from_previews
    monkeypatch.setattr(QueryBankGenerator, "generate_from_previews", lambda *args: (_ for _ in ()).throw(RuntimeError("service down")))
    with pytest.raises(SystemExit) as error:
        main(["all", *base])
    assert error.value.code == 1
    assert not (tmp_path / "output" / "scores.jsonl").exists()
    monkeypatch.setattr(QueryBankGenerator, "generate_from_previews", generate)
    main(["all", *base])
    assert counts["generate"] == 2


def test_validation_checks_expected_coverage_and_corrupt_arrays(tmp_path, monkeypatch):
    source, _, base = setup_pipeline(tmp_path, monkeypatch)
    main(["all", *base])
    args = build_parser().parse_args(["all", *base])
    path = pipeline.selection_path(args)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["selected_timestamps"][0] = float("nan")
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    args = build_parser().parse_args(["validate", "--input", str(source), "--selections", str(path), "--output-dir", str(tmp_path / "output")])
    assert pipeline.run_validate(args).failures == 2


def test_invalid_cli_parameters_fail_before_loading_models(tmp_path):
    with pytest.raises(SystemExit, match="fast-ratio"):
        main(["select", "--output-dir", str(tmp_path), "--fast-ratio", "nan"])
