import json

import pytest

from sportsqa_offline_sampling.artifacts import ConfigMismatchError, StageArtifact, prepare_jsonl_for_append
from sportsqa_offline_sampling.vision import FeatureCache


def test_prepare_jsonl_recovers_only_a_truncated_final_line(tmp_path):
    path = tmp_path / "output.jsonl"
    path.write_bytes(b'{"split":"train","qa_id":"1"}\n{"split"')
    records = prepare_jsonl_for_append(path)
    assert records == [{"split": "train", "qa_id": "1"}]
    assert path.read_text(encoding="utf-8") == '{"split":"train","qa_id":"1"}\n'


def test_stage_artifact_rejects_configuration_mismatch(tmp_path):
    output = tmp_path / "stage.jsonl"
    first = StageArtifact(output, tmp_path / "stage.config.json", {"setting": 1})
    first.prepare()
    with pytest.raises(ConfigMismatchError):
        StageArtifact(output, tmp_path / "stage.config.json", {"setting": 2}).prepare()


def test_feature_cache_rejects_a_changed_source_video(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"first-version")
    cache = FeatureCache(tmp_path / "cache", {"clip": "unit-test"})
    metadata = {"video_id": "sport/video", "candidate_frame_indices": [0, 1]}
    cache.save("sport/video", video, metadata, [[0.1, 0.2], [0.3, 0.4]])
    assert cache.load("sport/video", video, metadata).shape == (2, 2)
    video.write_bytes(b"changed-version")
    with pytest.raises(RuntimeError):
        cache.load("sport/video", video, metadata)
