#src/inference/timestamp_extractor.py
from typing import Tuple

SAMPLE_RATE = 16000
HOP_LENGTH = 160


def frame_to_time(frame_idx: int) -> float:
    return frame_idx * HOP_LENGTH / SAMPLE_RATE


def segment_frames_to_times(
    start_frame: int,
    end_frame: int,
) -> Tuple[float, float]:
    start_time = frame_to_time(start_frame)
    end_time = frame_to_time(end_frame + 1)
    return start_time, end_time
