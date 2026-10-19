import numpy as np

from sportsqa_offline_sampling.selection import adaptive_candidate_fps, candidate_frame_indices, select_slow_fast_frames


def test_candidate_fps_boundaries_and_final_frame():
    assert adaptive_candidate_fps(30.0) == 4.0
    assert adaptive_candidate_fps(30.01) == 2.0
    assert adaptive_candidate_fps(120.0) == 2.0
    assert adaptive_candidate_fps(120.01) == 1.0
    assert candidate_frame_indices(100, 10.0, 2.0) == [
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99
    ]


def test_selection_has_ordered_unique_frames_and_twelve_four_ratio():
    candidates = list(range(40))
    scores = np.asarray([
        [0.0] * 8 + [0.9, 1.0, 0.9, 0.8] + [0.0] * 28,
        [0.0] * 7 + [0.8, 0.9, 1.0, 0.9, 0.8] + [0.0] * 28,
    ])
    result = select_slow_fast_frames(scores, candidates, budget=16)
    assert len(result.frame_indices) == 16
    assert result.frame_indices == sorted(result.frame_indices)
    assert len(result.frame_indices) == len(set(result.frame_indices))
    assert result.sampling_types.count("slow") == 12
    assert result.sampling_types.count("fast") == 4


def test_flat_scores_and_short_candidate_pool_are_deterministic_and_ordered():
    flat_scores = [[0.25] * 9 for _ in range(7)]
    first = select_slow_fast_frames(flat_scores, list(range(0, 90, 10)), budget=16)
    second = select_slow_fast_frames(flat_scores, list(range(0, 90, 10)), budget=16)
    assert first == second
    assert first.frame_indices == list(range(0, 90, 10))
    assert first.sampling_types.count("slow") == 7
    assert first.sampling_types.count("fast") == 2
