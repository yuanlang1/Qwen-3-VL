"""Independent sports-video query generation and offline frame selection."""

from .consumer import decode_selected_frames, get_selection, load_selections
from .query_bank import QueryBankGenerator, select_query_prefix
from .selection import adaptive_candidate_fps, candidate_frame_indices, select_slow_fast_frames

PIPELINE_VERSION = "sports-video-sampling-v1"

__all__ = [
    "QueryBankGenerator", "select_query_prefix", "adaptive_candidate_fps",
    "candidate_frame_indices", "select_slow_fast_frames", "load_selections",
    "get_selection", "decode_selected_frames",
]
