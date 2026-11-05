import json

import pytest

from sports_video_sampling.records import VideoResolver, load_records


def test_standard_input_has_no_dataset_or_split_requirement(tmp_path):
    path = tmp_path / "any.jsonl"
    row = {"id": 1, "video": "match.mp4", "question": "Where is the ball?", "metadata": {"split": "challenge_round_2"}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    record = load_records(path)[0]
    assert record.id == "1"
    assert record.metadata["split"] == "challenge_round_2"


def test_duplicate_ids_are_rejected_even_beyond_smoke_limit(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps([{"id": value, "video": "v", "question": "q"} for value in (1, "1")]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path, limit=1)


def test_resolver_supports_exact_paths_and_explicit_mapping(tmp_path):
    video = tmp_path / "renamed.mp4"
    video.write_bytes(b"video")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"opaque-id": "renamed.mp4"}), encoding="utf-8")
    resolver = VideoResolver(tmp_path, index)
    assert resolver.resolve("opaque-id") == video
    assert resolver.resolve(str(video)) == video
    assert resolver.resolve("renamed") == video
    video.with_suffix(".avi").write_bytes(b"other-video")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolver.resolve("renamed")


@pytest.mark.parametrize("field,value", [("question", " "), ("video", None), ("id", True), ("metadata", [])])
def test_invalid_standard_records_are_rejected(tmp_path, field, value):
    row = {"id": "a", "video": "v", "question": "q", field: value}
    path = tmp_path / "input.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)
