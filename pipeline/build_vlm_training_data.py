#!/usr/bin/env python3
"""
Build VLM fine-tuning data from original caption data and Gemini rewritten data.
Multi-process version for faster image annotation.

Usage:
    python build_vlm_finetune_data_mp.py \
        --original /path/to/original.json \
        --rewritten /path/to/rewritten.json \
        --output /path/to/finetune.jsonl \
        --visualize-pose-dir /path/to/visualized_images \
        --num-workers 16
"""
from qwen_vl_utils import process_vision_info
import os
import json
import argparse
import hashlib
from typing import List, Dict, Optional, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm


# ================== System Prompt ==================

SYSTEM_PROMPT = """You are an expert image caption editor specializing in anatomical spatial accuracy (left/right consistency).
Your goal is to ensure descriptions of body parts match the image perfectly.

**CRITICAL CONCEPT: Perspective Alignment**
You must distinguish between 'Viewer's Side' (screen) and 'The Person's Own Side' (subject) to ensure accuracy.
- **Facing Camera:** Subject's Right hand is on Viewer's Left.


**Using Body Part Bounding Box Data**
You will be provided with body part bounding box annotations as a reference. The bbox data contains:
- **Body Part Names:** `left_*` and `right_*` refer to the subject's OWN anatomical sides (not viewer's perspective).
- **Bounding Box Format:** `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` where (x1,y1) is top-left and (x2,y2) is bottom-right, normalized to [0,1000].
- **Important Note:** The bounding box data contains ONLY PARTIAL body parts that were successfully detected and annotated. Not all body parts will have bounding boxes. Use the available bboxes as reference points, but rely on visual inspection for body parts without bbox annotations.
- **Detected Body Parts:** `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_hand`, `right_hand`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_foot`, `right_foot`.
- **Color Coding in Visualization:**
  - Green (left side): `left_*` body parts
  - Red (right side): `right_*` body parts


**Rules:**
1. **Exhaustive Verification:** You must identify and verify EVERY single body part mentioned in the original caption using both visual inspection and available bounding box data.
2. **Strict Scope:** Focus ONLY on human body parts already mentioned in the text. DO NOT change descriptions of objects, backgrounds, or sentence styles.
3. **Modification Criteria:**
   - **Case A: Error Correction:** If the caption explicitly states a side (left/right) that is visually INCORRECT based on the subject's own anatomy, you must CORRECT it.
   - **Case B: Conditional Clarification:** If a limb is mentioned WITHOUT a side, ONLY add 'left' or 'right' if it is necessary to distinguish between the two limbs (e.g., different actions or positions).
4. **Preservation:** Keep the original sentence structure and vocabulary exactly as is for all non-body-part content.
5. **Bounding Box Data as Reference:** Use available bounding box coordinates to assist judgment, but prioritize clear visual evidence when bbox data conflicts with what you see or when bbox data is unavailable.

"""

# ================== Data Loading Functions ==================

def load_original_data(input_path: str) -> List[Dict]:
    """Load original caption data (list format)."""
    print(f"Loading original data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'results' in data:
        return data['results']
    else:
        raise ValueError(f"Unsupported original data format in {input_path}")


def load_rewritten_data(input_path: str) -> Tuple[Dict, List[Dict]]:
    """Load rewritten data (dict format with meta and results)."""
    print(f"Loading rewritten data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        meta = data.get('meta', {})
        results = data.get('results', [])
        return meta, results
    elif isinstance(data, list):
        return {}, data
    else:
        raise ValueError(f"Unsupported rewritten data format in {input_path}")


def build_rewritten_index(rewritten_results: List[Dict]) -> Dict[int, Dict]:
    """Build index from rewritten results by idx."""
    index = {}
    for item in rewritten_results:
        if item is None:
            continue
        idx = item.get('idx')
        if idx is not None:
            index[idx] = item
    return index


def build_image_path_index(rewritten_results: List[Dict]) -> Dict[str, Dict]:
    """Build index from rewritten results by image_path."""
    index = {}
    for item in rewritten_results:
        if item is None:
            continue
        image_path = item.get('image_path')
        if image_path:
            index[image_path] = item
    return index


def build_caption_index(rewritten_results: List[Dict]) -> Dict[str, Dict]:
    """Build index from rewritten results by original_caption."""
    index = {}
    for item in rewritten_results:
        if item is None:
            continue
        original_caption = item.get('original_caption')
        if original_caption:
            index[original_caption] = item
    return index


# ================== Pose Visualization Functions ==================

def visualize_pose_on_image_worker(task: Dict) -> Dict:
    """
    Worker function for multiprocessing.
    在图像上绘制 Pose 关键点和标签，保存到新路径。
    
    Args:
        task: {
            'idx': int,
            'image_path': str,
            'pose_data': Dict,
            'output_path': str
        }
    
    Returns:
        {'idx': int, 'success': bool, 'output_path': str, 'error': str or None}
    """
    idx = task['idx']
    image_path = task['image_path']
    pose_data = task['pose_data']
    output_path = task['output_path']
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                'idx': idx,
                'success': False,
                'output_path': output_path,
                'error': f"Failed to load image: {image_path}"
            }
        
        # 如果没有有效的 pose 数据，直接保存原图
        if not pose_data or not pose_data.get('persons'):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            return {
                'idx': idx,
                'success': True,
                'output_path': output_path,
                'error': None
            }
        
        h, w, _ = img.shape
        font_scale = max(0.5, w / 1500.0)
        thickness = max(1, int(w / 800.0))
        circle_radius = max(3, int(w / 300.0))

        persons = pose_data.get('persons', [])
        for person in persons:
            keypoints = person.get('keypoints', {})
            if not keypoints:
                continue
                
            for kp_name, kp_info in keypoints.items():
                if kp_info is None:
                    continue
                
                x, y = int(kp_info['x']), int(kp_info['y'])
                
                # 颜色编码：红色=左侧(解剖学), 蓝色=右侧(解剖学), 绿色=中线
                if 'left' in kp_name:
                    color = (0, 0, 255)  # Red for LEFT anatomical side (BGR)
                    short_name = "L_" + kp_name.replace("left_", "")
                elif 'right' in kp_name:
                    color = (255, 0, 0)  # Blue for RIGHT anatomical side (BGR)
                    short_name = "R_" + kp_name.replace("right_", "")
                else:
                    color = (0, 255, 0)  # Green for midline
                    short_name = kp_name
                
                # 绘制空心圆
                cv2.circle(img, (x, y), circle_radius, color, thickness)
                
                # 绘制标签（带黑色描边）
                text_pos = (x + 10, y - 10)
                cv2.putText(img, short_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(img, short_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, thickness, cv2.LINE_AA)

        # 保存图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        
        return {
            'idx': idx,
            'success': True,
            'output_path': output_path,
            'error': None
        }
        
    except Exception as e:
        return {
            'idx': idx,
            'success': False,
            'output_path': output_path,
            'error': str(e)
        }


def get_visualized_image_path(original_path: str, output_dir: str) -> str:
    """
    根据原始图片路径生成可视化图片的保存路径。
    使用原始路径的 hash 作为子目录，避免文件名冲突。
    """
    filename = os.path.basename(original_path)
    dir_hash = hashlib.md5(os.path.dirname(original_path).encode()).hexdigest()[:8]
    output_path = os.path.join(output_dir, dir_hash, filename)
    return output_path


def batch_visualize_poses(
    tasks: List[Dict],
    num_workers: Optional[int] = None,
    desc: str = "Visualizing poses"
) -> Dict[int, Dict]:
    """
    批量并行处理图像可视化。
    
    Args:
        tasks: List of task dicts for visualize_pose_on_image_worker
        num_workers: Number of parallel workers
        desc: Progress bar description
    
    Returns:
        Dict mapping idx to result dict
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 100)
    
    results = {}
    
    if len(tasks) == 0:
        return results
    
    print(f"Starting pose visualization with {num_workers} workers...")
    
    # 使用 imap_unordered 以获得更好的性能
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(visualize_pose_on_image_worker, tasks, chunksize=32),
            total=len(tasks),
            desc=desc
        ):
            results[result['idx']] = result
    
    return results


# ================== Formatting Functions ==================

def normalize_bodypart_bbox_annotation(bodypart_annotation: Dict, target_range: int = 1000) -> Dict:
    """
    将 bodypart_bbox annotation 中的坐标归一化到 0~target_range 范围，并使用 <|box_start|> <|box_end|> 包裹坐标。
    
    Args:
        bodypart_annotation: 原始 bodypart_bbox annotation 数据
        target_range: 归一化目标范围 (默认 1000)
    
    Returns:
        归一化后的 bodypart_bbox annotation (深拷贝，不修改原数据)
    """
    import copy
    
    if not bodypart_annotation:
        return bodypart_annotation
    
    # 深拷贝，避免修改原数据
    normalized = copy.deepcopy(bodypart_annotation)
    
    # 获取图片尺寸
    image_size = normalized.get('image_size', 0)
    img_width = image_size.get('width', 0)
    img_height = image_size.get('height', 0)
    
    if img_width == 0 or img_height == 0:
        # 如果没有图片尺寸信息，无法归一化
        return normalized
    
    # 归一化 persons 中的 body part bboxes
    if 'persons' in normalized:
        for person in normalized['persons']:
            # 归一化 bbox 字段中的所有 body parts
            if 'bbox' in person and person['bbox']:
                normalized_bboxes = {}
                for part_name, bbox in person['bbox'].items():
                    if bbox and len(bbox) == 4:
                        normalized_bbox = [
                            round(bbox[0] / img_width * target_range, 1),
                            round(bbox[1] / img_height * target_range, 1),
                            round(bbox[2] / img_width * target_range, 1),
                            round(bbox[3] / img_height * target_range, 1)
                        ]
                        normalized_bboxes[part_name] = f"<|box_start|>({normalized_bbox[0]},{normalized_bbox[1]}),({normalized_bbox[2]},{normalized_bbox[3]})<|box_end|>"
                person['bbox'] = normalized_bboxes
    
    normalized.pop('image_size', None)
    return normalized


def format_bodypart_bbox_data(bodypart_annotation: Dict, normalize: bool = True, target_range: int = 1000) -> str:
    """
    Format bodypart_bbox annotation data as string.
    
    Args:
        bodypart_annotation: 原始 bodypart_bbox annotation 数据
        normalize: 是否归一化坐标 (默认 True)
        target_range: 归一化目标范围 (默认 1000)
    
    Returns:
        格式化后的 bodypart_bbox annotation 字符串
    """
    if not bodypart_annotation:
        return ""
    
    # 归一化坐标
    if normalize:
        bodypart_annotation = normalize_bodypart_bbox_annotation(bodypart_annotation, target_range)
    
    bbox_info = {}
    
    if 'image_size' in bodypart_annotation:
        bbox_info['image_size'] = bodypart_annotation['image_size']
    
    if 'persons' in bodypart_annotation:
        bbox_info['persons'] = bodypart_annotation['persons']
    
    if not bbox_info:
        bbox_info = bodypart_annotation
    
    return json.dumps(bbox_info, ensure_ascii=False, indent=2)


def format_assistant_response(
    corrected_caption: str,
    reasoning: str = "",
    modifications: Optional[List[str]] = None,
    include_reasoning: bool = True,
    include_modifications: bool = False
) -> str:
    """Format assistant response with optional reasoning and modifications."""
    parts = []
    
    if include_reasoning and reasoning:
        parts.append(f"Reasoning: {reasoning}")
    
    if include_modifications and modifications:
        mods_str = "\n".join(f"- {m}" for m in modifications)
        parts.append(f"Modifications:\n{mods_str}")
    
    parts.append(f"Corrected caption: {corrected_caption}")
    
    return "\n\n".join(parts)


# ================== Conversion Functions ==================

def build_single_output(
    original_item: Dict,
    rewritten_item: Dict,
    final_image_path: str,
    include_reasoning: bool = True,
    include_modifications: bool = False,
    include_bbox: bool = True
) -> Dict:
    """
    构建单个输出项（不包含图像处理）。
    使用 bodypart_bbox_annotation 而不是 pose_annotation。
    """
    original_caption = (
        rewritten_item.get('original_caption') or
        original_item.get('caption', '')
    )
    
    corrected_caption = rewritten_item.get('corrected_caption', original_caption)
    bodypart_annotation = original_item.get('bodypart_bbox_annotation', {})
    reasoning = rewritten_item.get('reasoning', '')
    modifications = rewritten_item.get('modifications', [])
    
    # Build user content
    user_parts = ["<image>", original_caption]
    
    if include_bbox:
        bbox_str = format_bodypart_bbox_data(bodypart_annotation)
        if bbox_str:
            user_parts.append(f"Body Part Bounding Boxes:\n{bbox_str}")
    
    user_content = "\n\n".join(user_parts)
    
    # Build assistant content
    assistant_content = format_assistant_response(
        corrected_caption=corrected_caption,
        reasoning=reasoning,
        modifications=modifications,
        include_reasoning=include_reasoning,
        include_modifications=include_modifications
    )
    
    # Build output format
    output = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [final_image_path]
    }
    
    return output


def convert_dataset_multiprocess(
    original_path: str,
    rewritten_path: str,
    json_output_path: str,
    only_modified: bool = False,
    only_success: bool = True,
    include_reasoning: bool = True,
    include_modifications: bool = False,
    include_bbox: bool = True,
    max_samples: Optional[int] = None,
    validate_images: bool = False,
    match_by: str = "idx",
    num_workers: Optional[int] = None
) -> Dict[str, int]:
    """
    多进程版本的数据集转换。
    
    Args:
        original_path: Path to original data JSON
        rewritten_path: Path to rewritten data JSON
        output_path: Output JSON/JSONL file path
        only_modified: Only keep samples with modifications
        only_success: Only keep successful samples
        include_reasoning: Include reasoning in output
        include_modifications: Include modifications list in output
        include_bbox: Include bodypart_bbox data in user prompt
        max_samples: Maximum number of samples
        validate_images: Validate image files exist
        match_by: How to match items ('idx', 'image_path', or 'caption')
        num_workers: Number of parallel workers for image processing
    
    Returns:
        Statistics dictionary
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 100)
    
    # Load data
    original_data = load_original_data(original_path)
    meta, rewritten_results = load_rewritten_data(rewritten_path)
    
    print(f"Original data count: {len(original_data)}")
    print(f"Rewritten data count: {len(rewritten_results)}")
    if meta:
        print(f"Rewritten meta: {meta}")
    
    # Build index for rewritten data
    rewritten_index_by_idx: Dict[int, Dict] = {}
    rewritten_index_by_path: Dict[str, Dict] = {}
    rewritten_index_by_caption: Dict[str, Dict] = {}
    
    if match_by == "idx":
        rewritten_index_by_idx = build_rewritten_index(rewritten_results)
        print(f"Built index by idx with {len(rewritten_index_by_idx)} items")
        # 调试：打印 rewritten_results 中前几个 item 的 idx 值
        if len(rewritten_results) > 0:
            sample_idxs = [item.get('idx') for item in rewritten_results[:5] if item]
            print(f"Sample idx values in rewritten data: {sample_idxs}")
            # 检查是否有 success 字段
            sample_success = [item.get('success') for item in rewritten_results[:5] if item]
            print(f"Sample success values in rewritten data: {sample_success}")
    elif match_by == "image_path":
        rewritten_index_by_path = build_image_path_index(rewritten_results)
        print(f"Built index by image_path with {len(rewritten_index_by_path)} items")
    elif match_by == "caption":
        rewritten_index_by_caption = build_caption_index(rewritten_results)
        print(f"Built index by caption (original_caption) with {len(rewritten_index_by_caption)} items")
        # 调试：打印前几个 original_caption 的样本
        if len(rewritten_results) > 0:
            sample_captions = [item.get('original_caption', '')[:50] + '...' for item in rewritten_results[:3] if item]
            print(f"Sample original_caption values in rewritten data: {sample_captions}")
    
    if max_samples:
        original_data = original_data[:max_samples]
    
    # Statistics
    stats = {
        'total': len(original_data),
        'matched': 0,
        'success': 0,
        'skipped_no_match': 0,
        'skipped_failed': 0,
        'skipped_no_modification': 0,
        'skipped_no_image': 0,
        'skipped_image_not_found': 0
    }
    
    # ========== Phase 1: Filter and prepare tasks ==========
    print("\n[Phase 1] Filtering and preparing data...")
    
    valid_items = []  # List of (idx, original_item, rewritten_item, image_path)
    
    for idx, original_item in enumerate(tqdm(original_data, desc="Filtering")):
        # Find matching rewritten item
        if match_by == "idx":
            rewritten_item = rewritten_index_by_idx.get(idx)
        elif match_by == "image_path":
            image_path = original_item.get('image_path')
            if image_path:
                rewritten_item = rewritten_index_by_path.get(image_path)
            else:
                rewritten_item = None
        elif match_by == "caption":
            # 使用 original 数据中的 caption 与 rewritten 数据中的 original_caption 进行匹配
            caption = original_item.get('caption')
            if caption:
                rewritten_item = rewritten_index_by_caption.get(caption)
            else:
                rewritten_item = None
        else:
            rewritten_item = None
        
        if rewritten_item is None:
            stats['skipped_no_match'] += 1
            continue
        
        stats['matched'] += 1
        
        if only_success and not rewritten_item.get('success', False):
            stats['skipped_failed'] += 1
            continue
        
        if only_modified and not rewritten_item.get('has_modification', False):
            stats['skipped_no_modification'] += 1
            continue
        
        image_path =  original_item.get('image_path')
        if not image_path:
            stats['skipped_no_image'] += 1
            continue
        
        if validate_images and not os.path.exists(image_path):
            stats['skipped_image_not_found'] += 1
            continue
        
        valid_items.append((idx, original_item, rewritten_item, image_path))
    
    print(f"Valid items after filtering: {len(valid_items)}")
    
    # ========== Phase 2: Build output data ==========
    print(f"\n[Phase 2] Building output data...")
    
    converted = []
    
    for idx, original_item, rewritten_item, image_path in tqdm(valid_items, desc="Building output"):
        # Use original image path directly (no visualization)
        final_image_path = image_path
        
        # Build output item
        output = build_single_output(
            original_item=original_item,
            rewritten_item=rewritten_item,
            final_image_path=final_image_path,
            include_reasoning=include_reasoning,
            include_modifications=include_modifications,
            include_bbox=include_bbox
        )
        
        converted.append(output)
        stats['success'] += 1
    
    # ========== Phase 3: Save results ==========
    print(f"\n[Phase 3] Saving {len(converted)} items to {json_output_path}...")
    output_dir = os.path.dirname(json_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if json_output_path.endswith('.jsonl'):
        with open(json_output_path, 'w', encoding='utf-8') as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
    
    return stats


def print_stats(stats: Dict[str, int]):
    """Print statistics."""
    print("\n" + "=" * 60)
    print("Conversion Statistics")
    print("=" * 60)
    print(f"Total original items:       {stats['total']}")
    print(f"Matched with rewritten:     {stats['matched']}")
    print(f"Successfully converted:     {stats['success']}")
    print("-" * 60)
    print(f"Skipped (no match):         {stats['skipped_no_match']}")
    print(f"Skipped (failed):           {stats['skipped_failed']}")
    print(f"Skipped (no modification):  {stats['skipped_no_modification']}")
    print(f"Skipped (no image path):    {stats['skipped_no_image']}")
    print(f"Skipped (image not found):  {stats['skipped_image_not_found']}")
    print("=" * 60)


# ================== CLI ==================

def main():
    parser = argparse.ArgumentParser(
        description='Build VLM fine-tuning data (Multi-process version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_vlm_finetune_data_mp.py \\
      --original /path/to/original.json \\
      --rewritten /path/to/rewritten.json \\
      --output /path/to/finetune.jsonl

  # Only keep modified samples
  python build_vlm_finetune_data_mp.py \\
      --original /path/to/original.json \\
      --rewritten /path/to/rewritten.json \\
      --output /path/to/finetune.jsonl \\
      --only-modified
        """
    )
    
    parser.add_argument(
        '--original', '-o',
        type=str,
        default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_sam_yolo_anno_dataset.json",
        help='Path to original caption data JSON'
    )
    
    parser.add_argument(
        '--rewritten', '-r',
        type=str,
        default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_dataset_eval.json",
        help='Path to rewritten (Gemini output) data JSON'
    )
    
    parser.add_argument(
        '--output', '-out',
        type=str,
        default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_bbox_sharegpt_eval.json",
        help='Output JSON/JSONL file path'
    )
    
    parser.add_argument(
        '--only-modified',
        action='store_true',
        help='Only keep samples with modifications'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include failed samples as well (default: only success)'
    )
    
    parser.add_argument(
        '--include-reasoning',
        action='store_true',
        help='Include reasoning in assistant response (default: True)'
    )
    
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        default=True,
        help='Exclude reasoning from assistant response'
    )
    
    parser.add_argument(
        '--include-modifications',
        action='store_true',
        help='Include modifications list in assistant response'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to convert'
    )
    
    parser.add_argument(
        '--validate-images',
        action='store_true',
        help='Validate that image files exist'
    )
    
    parser.add_argument(
        '--match-by',
        type=str,
        choices=['idx', 'image_path', 'caption'],
        default='caption',
        help='How to match original and rewritten items (default: idx)'
    )
    
    parser.add_argument(
        '--num-workers', '-j',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: min(cpu_count, 100))'
    )
    
    args = parser.parse_args()
    
    # Handle flag logic
    include_reasoning = not args.no_reasoning
    only_success = not args.include_all
    
    stats = convert_dataset_multiprocess(
        original_path=args.original,
        rewritten_path=args.rewritten,
        json_output_path=args.output,
        only_modified=args.only_modified,
        only_success=only_success,
        include_reasoning=include_reasoning,
        include_modifications=args.include_modifications,
        include_bbox=True,
        max_samples=args.max_samples,
        validate_images=args.validate_images,
        match_by=args.match_by,
        num_workers=args.num_workers
    )
    
    print_stats(stats)
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == '__main__':
    main()
