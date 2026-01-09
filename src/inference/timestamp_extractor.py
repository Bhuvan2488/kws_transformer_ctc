# src/inference/timestamp_extractor.py

from typing import List, Tuple, Dict


def frames_to_seconds(frame_idx: int, hop_length: int = 160, sample_rate: int = 16000) -> float:
    return frame_idx * hop_length / sample_rate


def extract_word_timestamps(
    char_spans: List[Tuple[int, int, int]],
    id2char: Dict[int, str],
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> List[Dict]:
    """
    Args:
        char_spans:
            [(char_id, start_frame, end_frame), ...]
        id2char:
            mapping from char_id to character
    Returns:
        [
          {
            "word": str,
            "start_time": float,
            "end_time": float
          },
          ...
        ]
    """

    words = []
    current_word = []
    word_start_frame = None
    word_end_frame = None

    for char_id, start_f, end_f in char_spans:
        ch = id2char[char_id]

        if ch == " ":
            if current_word:
                words.append({
                    "word": "".join(current_word),
                    "start_time": frames_to_seconds(word_start_frame, hop_length, sample_rate),
                    "end_time": frames_to_seconds(word_end_frame + 1, hop_length, sample_rate),
                })
                current_word = []
                word_start_frame = None
                word_end_frame = None
            continue

        if not current_word:
            word_start_frame = start_f

        current_word.append(ch)
        word_end_frame = end_f

    # flush last word
    if current_word:
        words.append({
            "word": "".join(current_word),
            "start_time": frames_to_seconds(word_start_frame, hop_length, sample_rate),
            "end_time": frames_to_seconds(word_end_frame + 1, hop_length, sample_rate),
        })

    return words
