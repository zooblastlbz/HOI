"""
MPI video pose annotation.

Consumes sampled video frames (images[0]) and runs YOLO-Pose per frame.
Outputs frame-level pose annotations and a video-level success summary.
"""

import argparse
import json
import math
import os
import shutil
import time
from glob import glob
from typing import Any, Dict, List, Tuple

from anno_pose import YoloPoseAnnotator, get_mpi_info


RANK, WORLD_SIZE = get_mpi_info()
LOCAL_RANK = RANK % 8 if RANK != -1 else 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)


def extract_frames(record: Dict[str, Any], image_key: str = "images", list_index: int = 0) -> List[str]:
    frames_container = record.get(image_key, [])
    if not isinstance(frames_container, list) or len(frames_container) <= list_index:
        return []
    frames = frames_container[list_index]
    return frames if isinstance(frames, list) else []


def annotate_videos(
    data_list: List[Dict[str, Any]],
    annotator: YoloPoseAnnotator,
    image_key: str = "images",
    list_index: int = 0,
    output_key: str = "pose_annotations",
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    """
    Run pose annotation for all frames across videos using batched inference.
    """
    path_to_indices: Dict[str, List[Tuple[int, int]]] = {}
    for vid_idx, item in enumerate(data_list):
        frames = extract_frames(item, image_key=image_key, list_index=list_index)
        for frame_pos, path in enumerate(frames):
            if not path:
                continue
            if path not in path_to_indices:
                path_to_indices[path] = []
            path_to_indices[path].append((vid_idx, frame_pos))

    unique_paths = list(path_to_indices.keys())
    print(f"[Rank {RANK}] Unique frames to process: {len(unique_paths)}")

    # Batched inference
    results_by_path: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(unique_paths), batch_size):
        end = min(start + batch_size, len(unique_paths))
        batch_paths = unique_paths[start:end]
        batch_results = annotator.process_batch(batch_paths)
        for p, res in zip(batch_paths, batch_results):
            results_by_path[p] = res
        print(f"[Rank {RANK}] Processed batch {start}-{end-1}")

    # Map back to videos
    for vid_idx, item in enumerate(data_list):
        frames = extract_frames(item, image_key=image_key, list_index=list_index)
        if not frames:
            item[output_key] = {"success": False, "error": "NO_FRAMES"}
            continue

        frame_results = []
        success_cnt = 0
        for pos, path in enumerate(frames):
            res = results_by_path.get(path, {"success": False, "error": "MISSING_RESULT"})
            wrapped = dict(res)
            wrapped["frame_idx_sampled"] = pos
            wrapped["frame_path"] = path
            frame_results.append(wrapped)
            if res.get("success"):
                success_cnt += 1

        item[output_key] = {
            "success": success_cnt == len(frames),
            "total_frames": len(frames),
            "success_frames": success_cnt,
            "frames": frame_results,
        }

    return data_list


def main():
    parser = argparse.ArgumentParser(description="Video Pose Annotation (MPI)")
    parser.add_argument("--model_path", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/yolo11x-pose.pt")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, default="./temp_pose_video_results")
    parser.add_argument("--image_key", type=str, default="images")
    parser.add_argument("--list_index", type=int, default=0)
    parser.add_argument("--output_key", type=str, default="pose_annotations")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--conf_threshold", type=float, default=0.6)
    args = parser.parse_args()

    if RANK == 0:
        print(f"[Pose Video] World size={WORLD_SIZE}")
        print(f"Input: {args.input_json}")
        print(f"Output: {args.output_json}")
        if not os.path.exists(args.input_json):
            print(f"Error: Input file {args.input_json} not found.")
            return

    with open(args.input_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    total_entries = len(all_data)
    samples_per_process = math.ceil(total_entries / WORLD_SIZE) if WORLD_SIZE > 0 else total_entries
    start_idx = RANK * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_entries)
    my_data = all_data[start_idx:end_idx]

    print(f"[Rank {RANK}] Processing {len(my_data)} samples (index {start_idx} to {end_idx})")

    if len(my_data) > 0:
        annotator = YoloPoseAnnotator(
            model_path=args.model_path,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
        )
        start_time = time.time()
        my_data = annotate_videos(
            my_data,
            annotator=annotator,
            image_key=args.image_key,
            list_index=args.list_index,
            output_key=args.output_key,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start_time
        print(f"[Rank {RANK}] Done in {elapsed:.2f}s")
    else:
        print(f"[Rank {RANK}] No data to process.")

    os.makedirs(args.temp_dir, exist_ok=True)
    temp_file = os.path.join(args.temp_dir, f"rank_{RANK}.json")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(my_data, f, ensure_ascii=False, indent=2)
    print(f"[Rank {RANK}] Saved temp results to {temp_file}")

    marker_dir = os.path.join(args.temp_dir, "markers")
    os.makedirs(marker_dir, exist_ok=True)
    marker_file = os.path.join(marker_dir, f"rank_{RANK}.done")
    with open(marker_file, "w") as f:
        f.write("done")

    if RANK == 0:
        print("[Rank 0] Waiting for all ranks...")
        while True:
            done_files = glob(os.path.join(marker_dir, "*.done"))
            if len(done_files) >= WORLD_SIZE:
                break
            time.sleep(2)

        final_data = []
        for r in range(WORLD_SIZE):
            t_file = os.path.join(args.temp_dir, f"rank_{r}.json")
            if os.path.exists(t_file):
                with open(t_file, "r", encoding="utf-8") as f:
                    final_data.extend(json.load(f))
            else:
                print(f"[Rank 0] Missing temp file from rank {r}")

        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        shutil.rmtree(args.temp_dir, ignore_errors=True)

        success_count = sum(1 for item in final_data if item.get(args.output_key, {}).get("success", False))
        print(f"[Rank 0] Success rate: {success_count}/{len(final_data)}")
        print(f"[Rank 0] Output saved to {args.output_json}")


if __name__ == "__main__":
    main()

