from types import SimpleNamespace

import numpy as np
import pytest

from sports_video_sampling.consumer import decode_selected_frames, load_selections
from sports_video_sampling import video_io


def selection():
    return {
        "id": "a", "video": "unused.mp4", "question": "Where is the ball?",
        "decoder": "decord", "source_video": {"total_frames": 10}, "candidate_count": 10,
        "selection": {"frame_budget": 3}, "actual_frame_count": 3,
        "selected_frame_indices": [1, 4, 9], "selected_timestamps": [0.1, 0.4, 0.9],
        "sampling_types": ["slow", "fast", "slow"], "clip_ids": [0, -1, 1],
        "relevance_scores": [0.7, 0.1, 0.9],
    }


def test_consumer_decodes_exact_indices_with_the_selection_backend(monkeypatch):
    class Reader:
        def __len__(self):
            return 10

        def get_batch(self, indices):
            assert indices == [1, 4, 9]
            return SimpleNamespace(asnumpy=lambda: np.asarray(indices)[:, None, None, None])

        def get_frame_timestamp(self, indices):
            return [[index / 10, (index + 1) / 10] for index in indices]

        def seek(self, index):
            pass

    monkeypatch.setattr(video_io, "_video_reader", lambda path: Reader())
    result = decode_selected_frames("unused.mp4", selection())
    assert result["frames"].flatten().tolist() == [1, 4, 9]
    assert result["timestamps"] == [0.1, 0.4, 0.9]
    changed = selection()
    changed["selected_timestamps"][0] = 0.2
    with pytest.raises(ValueError, match="timestamps"):
        decode_selected_frames("unused.mp4", changed)


def test_consumer_rejects_duplicates_before_decoding():
    changed = selection()
    changed["selected_frame_indices"] = [1, 1, 9]
    with pytest.raises(ValueError, match="sorted and unique"):
        decode_selected_frames("unused.mp4", changed)


def test_missing_selection_file_is_not_reported_as_an_empty_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_selections(tmp_path / "missing.jsonl")


def test_preview_is_chronological_and_bounded(monkeypatch, tmp_path):
    calls = []

    class Reader:
        def __len__(self):
            return 16

        def get_batch(self, indices):
            calls.append(indices)
            return SimpleNamespace(asnumpy=lambda: np.zeros((len(indices), 80, 160, 3), dtype=np.uint8))

        def seek(self, index):
            pass

    monkeypatch.setattr(video_io, "_video_reader", lambda path: Reader())
    images, indices = video_io.decode_preview_frames(tmp_path / "unused.mp4", 4, 32)
    assert calls == [[0, 5, 10, 15]]
    assert indices == [0, 5, 10, 15]
    assert [image.size for image in images] == [(32, 16)] * 4
