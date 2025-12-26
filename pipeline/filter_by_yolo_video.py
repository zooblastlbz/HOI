"""
Video-aware YOLO filtering with frame sampling.

Key behaviors:
 1) Downsample frames to target_fps (original_fps defaults to 25).
 2) If ANY sampled frame passes the person-count/conf threshold, keep the whole video.
 3) Persist only the sampled frames in images[0], along with sampling metadata and per-frame YOLO results.
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from filter_by_yolo import YOLOFilter, get_mpi_info, load_json_records, wrap_json_records
from video_utils import sample_frame_paths


RANK, WORLD_SIZE = get_mpi_info()
LOCAL_RANK = RANK % 8 if RANK != -1 else 0


def extract_frames(
    record: Dict[str, Any], image_key: str = "images", list_index: int = 0
) -> List[str]:
    """
    Get the frame list from record[image_key][list_index] safely.
    """
    frames_container = record.get(image_key, [])
    if not isinstance(frames_container, list) or len(frames_container) <= list_index:
        return []
    frames = frames_container[list_index]
    return frames if isinstance(frames, list) else []


def filter_video_record(
    record: Dict[str, Any],
    yolo_filter: YOLOFilter,
    target_fps: float,
    original_fps: float,
    max_frames: Optional[int],
    image_key: str = "images",
    list_index: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Run sampling + YOLO filter for a single video record.
    Returns (kept_record_or_None, stats_dict_for_record).
    """
    frames = extract_frames(record, image_key=image_key, list_index=list_index)
    per_record_stats = {
        "total_frames": len(frames),
        "sampled_frames": 0,
        "valid_frames": 0,
        "kept": False,
        "reason": "",
    }
    if not frames:
        per_record_stats["reason"] = "NO_FRAMES"
        return None, per_record_stats

    sampled_frames, sampling_meta = sample_frame_paths(
        frames, original_fps=original_fps, target_fps=target_fps, max_frames=max_frames
    )
    per_record_stats["sampled_frames"] = len(sampled_frames)

    if not sampled_frames:
        per_record_stats["reason"] = "NO_SAMPLED_FRAMES"
        return None, per_record_stats

    # YOLO inference on sampled frames
    batch_results = yolo_filter.process_batch(sampled_frames)

    frame_results: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    for i, (is_valid, person_count, confidences) in enumerate(batch_results):
        frame_idx_orig = sampling_meta["frame_indices"][i] if i < len(sampling_meta.get("frame_indices", [])) else i
        frame_info = {
            "frame_idx_sampled": i,
            "frame_idx_orig": frame_idx_orig,
            "frame_path": sampled_frames[i],
            "person_count": int(person_count) if person_count is not None else 0,
            "person_confidences": [float(c) for c in confidences],
            "valid": bool(is_valid),
        }
        frame_results.append(frame_info)
        if is_valid:
            valid_indices.append(i)

    per_record_stats["valid_frames"] = len(valid_indices)

    if not valid_indices:
        per_record_stats["reason"] = "NO_VALID_FRAME"
        return None, per_record_stats

    kept = dict(record)
    # Replace images with sampled frames only
    kept[image_key] = [sampled_frames]
    kept["sampling_meta"] = sampling_meta
    kept["yolo_filter"] = {
        "frame_results": frame_results,
        "valid": True,
        "valid_frame_indices": valid_indices,
    }

    per_record_stats["kept"] = True
    per_record_stats["reason"] = "HAS_VALID_FRAME"
    return kept, per_record_stats


def process_video_records(
    records: List[Dict[str, Any]],
    yolo_filter: YOLOFilter,
    target_fps: float,
    original_fps: float,
    max_frames: Optional[int],
    image_key: str = "images",
    list_index: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter a list of video records.
    """
    kept_records: List[Dict[str, Any]] = []
    stats = {"total_processed": 0, "kept": 0, "dropped": 0}

    for record in records:
        kept, rec_stats = filter_video_record(
            record,
            yolo_filter=yolo_filter,
            target_fps=target_fps,
            original_fps=original_fps,
            max_frames=max_frames,
            image_key=image_key,
            list_index=list_index,
        )
        stats["total_processed"] += 1
        if kept is not None:
            kept_records.append(kept)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1
    return kept_records, stats


def main():
    parser = argparse.ArgumentParser(description="Video YOLO filter with frame sampling (JSON input)")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file (list or dict with results)")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file for filtered data")
    parser.add_argument("--image_key", type=str, default="images", help="JSON key for frames container")
    parser.add_argument("--list_index", type=int, default=0, help="Index inside the images list to use")
    parser.add_argument("--target_fps", type=float, default=5.0, help="Target sampling FPS")
    parser.add_argument("--original_fps", type=float, default=25.0, help="Original FPS of frames")
    parser.add_argument("--max_frames_per_video", type=int, default=None, help="Optional cap after sampling")
    parser.add_argument("--model", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/yolo12x.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_persons", type=int, default=1)
    parser.add_argument("--max_persons", type=int, default=5)
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=None, help="Optional subset for debugging")
    parser.add_argument("--temp_dir", type=str, default="./temp_video_yolo_filter", help="Temporary dir for MPI merge")
    args = parser.parse_args()

    if RANK == 0:
        print(f"[Video YOLO Filter] World size={WORLD_SIZE}")
        print(f"Input: {args.input_json}")
        print(f"Output: {args.output_json}")
        if not os.path.exists(args.input_json):
            print(f"Error: Input file {args.input_json} not found.")
            return

    records, wrapper_info = load_json_records(args.input_json)
    if args.num_samples:
        records = records[: args.num_samples]

    total_entries = len(records)
    samples_per_process = math.ceil(total_entries / WORLD_SIZE) if WORLD_SIZE > 0 else total_entries
    start_idx = RANK * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_entries)
    my_records = records[start_idx:end_idx]

    if RANK == 0:
        print(f"Total entries: {total_entries}, Per process: ~{samples_per_process}")
    print(f"[Rank {RANK}] Processing {len(my_records)} samples (index {start_idx} to {end_idx})")

    # Init YOLO filter per rank
    yolo_filter = YOLOFilter(
        model_path=args.model,
        gpu_id=LOCAL_RANK,
        min_persons=args.min_persons,
        max_persons=args.max_persons,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
    )

    filtered_records, stats = process_video_records(
        my_records,
        yolo_filter=yolo_filter,
        target_fps=args.target_fps,
        original_fps=args.original_fps,
        max_frames=args.max_frames_per_video,
        image_key=args.image_key,
        list_index=args.list_index,
    )

    os.makedirs(args.temp_dir, exist_ok=True)
    temp_output = os.path.join(args.temp_dir, f"video_yolo_filter_rank{RANK}.json")
    with open(temp_output, "w", encoding="utf-8") as f:
        json.dump(filtered_records, f, ensure_ascii=False, indent=2)

    marker_dir = os.path.join(args.temp_dir, "markers")
    os.makedirs(marker_dir, exist_ok=True)
    with open(os.path.join(marker_dir, f"rank_{RANK}.done"), "w") as f:
        f.write("done")

    print(f"[Rank {RANK}] Stats: {stats}")
    print(f"[Rank {RANK}] Saved temp output: {temp_output}")

    if RANK == 0:
        # wait for other ranks
        while True:
            done_files = glob(os.path.join(marker_dir, "*.done"))
            if len(done_files) >= WORLD_SIZE:
                break
            time.sleep(1)

        merged: List[Dict[str, Any]] = []
        for r in range(WORLD_SIZE):
            path = os.path.join(args.temp_dir, f"video_yolo_filter_rank{r}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))

        output_obj = wrap_json_records(merged, wrapper_info)
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)

        shutil.rmtree(args.temp_dir, ignore_errors=True)
        print(f"[Rank 0] Done. Kept {len(merged)}/{total_entries}. Output -> {args.output_json}")


if __name__ == "__main__":
    main()

