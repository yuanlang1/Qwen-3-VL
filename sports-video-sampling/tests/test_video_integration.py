"""Optional real Decord check; requires the video extra and FFmpeg executable."""

import shutil
import subprocess

import numpy as np
import pytest

from sports_video_sampling.video_io import decode_frame_indices, inspect_video, read_frame_timestamps


def test_real_video_exact_frame_decode(tmp_path):
    pytest.importorskip("decord")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg executable is required to create the test video")
    path = tmp_path / "frames.avi"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=64x48:rate=10:duration=1", "-c:v", "mpeg4", "-q:v", "2", str(path),
    ], check=True, capture_output=True)
    info = inspect_video(path)
    assert info.total_frames == 10
    all_frames, _, _ = decode_frame_indices(path, list(range(10)))
    frames, timestamps, total = decode_frame_indices(path, [1, 4, 9])
    assert total == 10
    np.testing.assert_array_equal(frames, all_frames[[1, 4, 9]])
    np.testing.assert_allclose(timestamps, read_frame_timestamps(path, [1, 4, 9]), atol=0.001)
