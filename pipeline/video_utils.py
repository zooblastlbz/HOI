import math
import os
import re
from typing import Dict, List, Tuple


_FRAME_NUM_PATTERN = re.compile(r"(\d+)(?=\.[^.]+$)")


def _frame_sort_key(path: str) -> int:
    """
    Extract a numeric index from a frame filename like 000123.jpg.
    Falls back to 0 if no digits are found.
    """
    filename = os.path.basename(path)
    m = _FRAME_NUM_PATTERN.search(filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    return 0


def compute_sampling_indices(
    total_frames: int,
    original_fps: float = 25.0,
    target_fps: float = 5.0,
    max_frames: int = None,
    force_last: bool = True,
) -> List[int]:
    """
    Compute frame indices for uniform downsampling.
    """
    if total_frames <= 0:
        return []
    if target_fps <= 0 or original_fps <= 0:
        return list(range(total_frames))

    stride = max(1, round(original_fps / target_fps))
    indices = list(range(0, total_frames, stride))
    if force_last and indices and indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)

    if max_frames is not None and len(indices) > max_frames:
        keep = max_frames
        indices = indices[:keep]

    return indices


def sample_frame_paths(
    frames: List[str],
    original_fps: float = 25.0,
    target_fps: float = 5.0,
    max_frames: int = None,
) -> Tuple[List[str], Dict]:
    """
    Downsample frame list and return sampled paths with metadata.
    """
    # Ensure frames are ordered by their numeric suffix (e.g., 000001.jpg).
    frames = sorted(frames, key=_frame_sort_key)
    total = len(frames)
    idxs = compute_sampling_indices(total, original_fps, target_fps, max_frames)
    sampled = [frames[i] for i in idxs if 0 <= i < total]
    meta = {
        "original_fps": original_fps,
        "target_fps": target_fps,
        "stride": max(1, round(original_fps / target_fps)) if target_fps > 0 else 1,
        "frame_indices": idxs,
        "total_frames": total,
        "sampled_frames": len(sampled),
        "max_frames": max_frames,
    }
    return sampled, meta
