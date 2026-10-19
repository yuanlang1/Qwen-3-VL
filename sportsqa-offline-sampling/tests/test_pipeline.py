import json
from pathlib import Path
from types import SimpleNamespace

from sportsqa_offline_sampling.pipeline import _paths, run_select, run_validate


def _score_record(qa_id, question, peak):
    candidates = list(range(32))
    scores = []
    for query in range(7):
        row = [0.0] * 32
        row[peak] = 1.0
        row[min(31, peak + query % 3)] = 0.9
        scores.append(row)
    return {
        "split": "train",
        "qa_id": str(qa_id),
        "video": "basketball/demo",
        "resolved_video": "basketball/demo.mp4",
        "question": question,
        "query_bank": [f"query {index}" for index in range(7)],
        "source_video": {"total_frames": 32, "average_fps": 10.0},
        "candidate_frames": {
            "fps": 4.0,
            "frame_indices": candidates,
            "timestamps": [index / 10 for index in candidates],
        },
        "query_frame_scores": scores,
    }


def _args(metadata_root: Path, output_dir: Path):
    return SimpleNamespace(
        metadata_root=metadata_root,
        video_root=output_dir / "videos-not-needed-for-structural-validation",
        output_dir=output_dir,
        splits=["train"],
        limit=None,
        num_queries=7,
        frame_budget=16,
        overwrite=False,
        check_videos=False,
    )


def test_selection_is_per_qa_resumable_and_validated(tmp_path):
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    metadata_root.joinpath("train.json").write_text(json.dumps([
        {"qa_id": 1, "video": "basketball/demo", "question": "What does the player do?"},
        {"qa_id": 2, "video": "basketball/demo", "question": "Where is the ball?"},
    ]), encoding="utf-8")
    output_dir = tmp_path / "output"
    paths = _paths(output_dir, "train")
    paths["scores"].parent.mkdir(parents=True)
    paths["scores"].write_text(
        "\n".join(json.dumps(record) for record in (
            _score_record(1, "What does the player do?", 5),
            _score_record(2, "Where is the ball?", 24),
        )) + "\n",
        encoding="utf-8",
    )
    args = _args(metadata_root, output_dir)

    summary = run_select(args)[0]
    assert summary.processed == 2
    selected = [json.loads(line) for line in _paths(output_dir, "train")["selection"].read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 2
    assert all(row["actual_frame_count"] == 16 for row in selected)
    assert all(row["selected_frame_indices"] == sorted(row["selected_frame_indices"]) for row in selected)
    assert selected[0]["selected_frame_indices"] != selected[1]["selected_frame_indices"]

    resumed = run_select(args)[0]
    assert resumed.processed == 0
    assert resumed.skipped == 2
    validation = run_validate(args)[0]
    assert validation.failures == 0
