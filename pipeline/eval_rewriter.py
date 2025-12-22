#!/usr/bin/env python3
"""
Batch offline inference for fine-tuned Qwen3-VL model using vLLM.

Features:
1. Efficient batch inference with vLLM
2. Checkpoint/resume support for large datasets
3. Configurable batch size to manage memory
4. Progress saving and recovery

Usage:
    python evaluate_finetuned_model.py \
        --input-data /path/to/eval_dataset.json \
        --model-path /path/to/finetuned_model \
        --output-dir /path/to/output \
        --batch-size 32
"""

import os
import json
import argparse
import random
import shutil
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Set multiprocessing method for vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import torch
from PIL import Image
from tqdm import tqdm


# ================== System Prompt (same as training) ==================

SYSTEM_PROMPT = """You are an expert image caption editor specializing in anatomical spatial accuracy (left/right consistency).
Your goal is to ensure descriptions of body parts match the image perfectly.

**CRITICAL CONCEPT: Perspective Alignment**
You must distinguish between 'Viewer's Side' (screen) and 'The Person's Own Side' (subject) to ensure accuracy.
- **Facing Camera:** Subject's Right hand is on Viewer's Left.


**Using Pose Annotation Data**
You will be provided with pose keypoint annotations as a reference. The pose data contains:
- **Keypoint Names:** `left_*` and `right_*` refer to the subject's OWN anatomical sides (not viewer's perspective).
- **Keypoint Coordinates:** (x, y) pixel positions in the image.
- **Common Keypoints:** `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`.


**Rules:**
1. **Exhaustive Verification:** You must identify and verify EVERY single body part mentioned in the original caption using both visual inspection and pose data.
2. **Strict Scope:** Focus ONLY on human body parts already mentioned in the text. DO NOT change descriptions of objects, backgrounds, or sentence styles.
3. **Modification Criteria:**
   - **Case A: Error Correction:** If the caption explicitly states a side (left/right) that is visually INCORRECT based on the subject's own anatomy, you must CORRECT it.
   - **Case B: Conditional Clarification:** If a limb is mentioned WITHOUT a side, ONLY add 'left' or 'right' if it is necessary to distinguish between the two limbs (e.g., different actions or positions).
4. **Preservation:** Keep the original sentence structure and vocabulary exactly as is for all non-body-part content.
5. **Pose Data as Reference:** Use pose keypoints to assist judgment, but prioritize clear visual evidence when pose data conflicts with what you see.

"""


# ================== Caption Parsing Functions ==================

def extract_original_caption(user_content: str) -> str:
    """
    从 user content 中提取原始 caption。
    格式: <image>\n\n{caption}\n\nPose Annotation:\n{pose_data}
    """
    # 移除 <image> 标签
    text = user_content.replace('<image>', '').strip()
    
    # 按 "Pose Annotation:" 分割，取前面的部分
    if 'Pose Annotation:' in text:
        caption = text.split('Pose Annotation:')[0].strip()
    else:
        caption = text.strip()
    
    return caption


def extract_corrected_caption(assistant_content: str) -> Tuple[str, str]:
    """
    从 assistant content 中提取 reasoning 和 corrected caption。
    格式: Reasoning: {reasoning}\n\nCorrected caption: {caption}
    
    Returns:
        (reasoning, corrected_caption)
    """
    reasoning = ""
    corrected_caption = ""
    
    # 提取 Reasoning
    if 'Reasoning:' in assistant_content:
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\n\nCorrected caption:|$)', assistant_content, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
    
    # 提取 Corrected caption
    if 'Corrected caption:' in assistant_content:
        caption_match = re.search(r'Corrected caption:\s*(.*)', assistant_content, re.DOTALL)
        if caption_match:
            corrected_caption = caption_match.group(1).strip()
    else:
        # 如果没有 "Corrected caption:" 标记，整个内容可能就是 caption
        corrected_caption = assistant_content.strip()
    
    return reasoning, corrected_caption


def extract_pose_annotation(user_content: str) -> str:
    """从 user content 中提取 pose annotation。"""
    if 'Pose Annotation:' in user_content:
        pose_str = user_content.split('Pose Annotation:')[1].strip()
        return pose_str
    return ""


def parse_training_item(item: Dict) -> Dict:
    """
    解析训练数据格式的 item。
    
    Returns:
        {
            'image_path': str,
            'original_caption': str,
            'ground_truth_caption': str,
            'ground_truth_reasoning': str,
            'pose_annotation': str,
            'user_content': str  # 完整的 user content，用于推理
        }
    """
    messages = item.get('messages', [])
    images = item.get('images', [])
    
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user':
            user_content = content
        elif role == 'assistant':
            assistant_content = content
    
    # 提取各部分
    original_caption = extract_original_caption(user_content)
    gt_reasoning, gt_caption = extract_corrected_caption(assistant_content)
    pose_annotation = extract_pose_annotation(user_content)
    
    image_path = images[0] if images else ""
    
    return {
        'image_path': image_path,
        'original_caption': original_caption,
        'ground_truth_caption': gt_caption,
        'ground_truth_reasoning': gt_reasoning,
        'pose_annotation': pose_annotation,
        'user_content': user_content
    }


# ================== Diff Analysis Functions ==================

def compute_text_diff(text1: str, text2: str) -> Dict:
    """
    计算两个文本之间的差异。
    
    Returns:
        {
            'is_identical': bool,
            'diff_ratio': float,  # 相似度 0-1
            'changes': List[Dict],  # 具体变化列表
            'summary': str  # 差异摘要
        }
    """
    import difflib
    
    if text1.strip() == text2.strip():
        return {
            'is_identical': True,
            'diff_ratio': 1.0,
            'changes': [],
            'summary': 'No changes'
        }
    
    # 计算相似度
    seq_matcher = difflib.SequenceMatcher(None, text1, text2)
    diff_ratio = seq_matcher.ratio()
    
    # 获取具体变化（基于单词级别）
    words1 = text1.split()
    words2 = text2.split()
    
    word_matcher = difflib.SequenceMatcher(None, words1, words2)
    changes = []
    
    for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            changes.append({
                'type': 'replace',
                'original': ' '.join(words1[i1:i2]),
                'new': ' '.join(words2[j1:j2]),
                'position': i1
            })
        elif tag == 'delete':
            changes.append({
                'type': 'delete',
                'original': ' '.join(words1[i1:i2]),
                'new': '',
                'position': i1
            })
        elif tag == 'insert':
            changes.append({
                'type': 'insert',
                'original': '',
                'new': ' '.join(words2[j1:j2]),
                'position': i1
            })
    
    # 生成摘要
    if not changes:
        summary = 'Whitespace or formatting changes only'
    else:
        change_types = [c['type'] for c in changes]
        summary = f"{len(changes)} changes: {change_types.count('replace')} replacements, {change_types.count('delete')} deletions, {change_types.count('insert')} insertions"
    
    return {
        'is_identical': False,
        'diff_ratio': diff_ratio,
        'changes': changes,
        'summary': summary
    }


def compute_caption_diffs(original: str, ground_truth: str, predicted: str) -> Dict:
    """
    计算三个 caption 之间的差异。
    
    Returns:
        {
            'original_vs_ground_truth': {...},  # 原始 vs GT (期望的修正)
            'original_vs_predicted': {...},     # 原始 vs 预测 (模型实际修正)
            'ground_truth_vs_predicted': {...}, # GT vs 预测 (模型与标准答案的差异)
            'analysis': str  # 综合分析
        }
    """
    diff_orig_gt = compute_text_diff(original, ground_truth)
    diff_orig_pred = compute_text_diff(original, predicted)
    diff_gt_pred = compute_text_diff(ground_truth, predicted)
    
    # 综合分析
    analysis_parts = []
    
    # 1. 检查 GT 是否有修改
    if diff_orig_gt['is_identical']:
        analysis_parts.append("Ground truth has NO modifications to original caption.")
    else:
        analysis_parts.append(f"Ground truth modified original: {diff_orig_gt['summary']}")
    
    # 2. 检查模型是否有修改
    if diff_orig_pred['is_identical']:
        analysis_parts.append("Model made NO modifications to original caption.")
    else:
        analysis_parts.append(f"Model modified original: {diff_orig_pred['summary']}")
    
    # 3. 检查模型与 GT 是否一致
    if diff_gt_pred['is_identical']:
        analysis_parts.append("✓ Model output MATCHES ground truth exactly.")
    else:
        analysis_parts.append(f"✗ Model output DIFFERS from ground truth: {diff_gt_pred['summary']}")
    
    return {
        'original_vs_ground_truth': diff_orig_gt,
        'original_vs_predicted': diff_orig_pred,
        'ground_truth_vs_predicted': diff_gt_pred,
        'analysis': ' | '.join(analysis_parts)
    }


def format_diff_for_display(diff_result: Dict) -> str:
    """将 diff 结果格式化为可读字符串。"""
    lines = []
    
    if diff_result['is_identical']:
        return "No differences"
    
    lines.append(f"Similarity: {diff_result['diff_ratio']:.2%}")
    
    for change in diff_result['changes'][:10]:  # 最多显示10个变化
        if change['type'] == 'replace':
            lines.append(f"  REPLACE: '{change['original']}' → '{change['new']}'")
        elif change['type'] == 'delete':
            lines.append(f"  DELETE: '{change['original']}'")
        elif change['type'] == 'insert':
            lines.append(f"  INSERT: '{change['new']}'")
    
    if len(diff_result['changes']) > 10:
        lines.append(f"  ... and {len(diff_result['changes']) - 10} more changes")
    
    return '\n'.join(lines)


# ================== Data Loading ==================

def load_input_data(input_path: str) -> List[Dict]:
    """Load input data (JSON or JSONL format)."""
    print(f"Loading data from {input_path}...")
    
    data = []
    if input_path.endswith('.jsonl'):
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            elif isinstance(loaded, dict) and 'results' in loaded:
                data = loaded['results']
            elif isinstance(loaded, dict) and 'data' in loaded:
                data = loaded['data']
            else:
                data = [loaded]
    
    print(f"Loaded {len(data)} samples")
    return data


def sample_data(data: List[Dict], num_samples: int, seed: Optional[int] = None) -> List[Dict]:
    """Randomly sample data."""
    if num_samples <= 0 or num_samples >= len(data):
        print(f"Using all {len(data)} samples")
        return data
    
    if seed is not None:
        random.seed(seed)
    
    sampled = random.sample(data, num_samples)
    print(f"Sampled {len(sampled)} items from {len(data)} total")
    return sampled


# ================== Checkpoint Management ==================

def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {checkpoint['processed_count']} samples already processed")
        return checkpoint
    return {"processed_count": 0, "results": []}


def save_checkpoint(checkpoint_path: str, results: List[Dict], processed_count: int):
    """Save checkpoint."""
    checkpoint = {
        "processed_count": processed_count,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# ================== vLLM Qwen3-VL Inference Engine ==================

def prepare_inputs_for_vllm(messages: List[Dict], processor) -> Dict:
    """Prepare inputs for vLLM Qwen3-VL inference."""
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
   
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data
    }


class Qwen3VLvLLMEngine:
    """vLLM-based inference engine for Qwen3-VL models."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = None,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        limit_mm_per_prompt: Optional[Dict] = None
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size or torch.cuda.device_count()
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 10, "video": 10}
        
        self.llm = None
        self.processor = None
    
    def load_model(self):
        """Load Qwen3-VL model using vLLM."""
        from vllm import LLM
        from transformers import AutoProcessor
        
        print(f"Loading Qwen3-VL model from {self.model_path} using vLLM...")
        print(f"  - tensor_parallel_size: {self.tensor_parallel_size}")
        print(f"  - max_model_len: {self.max_model_len}")
        print(f"  - gpu_memory_utilization: {self.gpu_memory_utilization}")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            trust_remote_code=True,
            seed=0
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def build_messages(self, image_path: str, user_text: str) -> List[Dict]:
        """Build chat messages for Qwen3-VL model."""
        clean_user_text = user_text.replace('<image>', '').strip()
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": clean_user_text}
                ]
            }
        ]
        
        return messages
    
    def batch_inference(
        self,
        samples: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_k: int = -1
    ) -> List[str]:
        """Run batch inference on multiple samples."""
        from vllm import SamplingParams
        
        # Build all inputs
        inputs = []
        valid_indices = []
        
        for i, sample in enumerate(samples):
            image_path = sample['image_path']
            user_text = sample['user_text']
            
            if not os.path.exists(image_path):
                continue
            
            try:
                messages = self.build_messages(image_path, user_text)
                vllm_input = prepare_inputs_for_vllm(messages, self.processor)
                inputs.append(vllm_input)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error building prompt for {image_path}: {e}")
                continue
        
        if not inputs:
            return ["[ERROR] Failed to process"] * len(samples)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop_token_ids=[]
        )
        
        # Run batch inference
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        
        # Map outputs back to original order
        results = ["[ERROR] Failed to process"] * len(samples)
        for idx, output in zip(valid_indices, outputs):
            generated_text = output.outputs[0].text
            results[idx] = generated_text
        
        return results


# ================== Image Saving ==================

def save_image(src_path: str, dst_dir: str, index: int) -> str:
    """Copy image to destination directory with index prefix."""
    if not os.path.exists(src_path):
        return ""
    
    os.makedirs(dst_dir, exist_ok=True)
    
    ext = os.path.splitext(src_path)[1]
    filename = f"{index:06d}{ext}"
    dst_path = os.path.join(dst_dir, filename)
    
    shutil.copy2(src_path, dst_path)
    return dst_path


# ================== Batch Processing ==================

def process_batch(
    batch_data: List[Dict],
    batch_indices: List[int],
    engine: Qwen3VLvLLMEngine,
    image_dir: Optional[str],
    max_tokens: int = 1024
) -> List[Dict]:
    """Process a batch of samples."""
    
    # Prepare samples for inference
    prepared_samples = []
    sample_info = []
    
    for idx, item in zip(batch_indices, batch_data):
        # 解析训练数据格式
        parsed = parse_training_item(item)
        
        image_path = parsed['image_path']
        user_content = parsed['user_content']
        
        # Save image if needed
        saved_image_path = ""
        if image_dir and image_path and os.path.exists(image_path):
            saved_image_path = save_image(image_path, image_dir, idx)
        
        prepared_samples.append({
            "image_path": image_path,
            "user_text": user_content
        })
        
        sample_info.append({
            "idx": idx,
            "image_path": image_path,
            "saved_image_path": saved_image_path,
            "original_caption": parsed['original_caption'],
            "ground_truth_caption": parsed['ground_truth_caption'],
            "ground_truth_reasoning": parsed['ground_truth_reasoning'],
            "pose_annotation": parsed['pose_annotation']
        })
    
    # Run batch inference
    predictions = engine.batch_inference(prepared_samples, max_tokens=max_tokens)
    
    # Build results
    results = []
    for info, prediction in zip(sample_info, predictions):
        # 解析模型输出
        pred_reasoning, pred_caption = extract_corrected_caption(prediction)
        
        # 计算三个 caption 之间的差异
        diffs = compute_caption_diffs(
            info['original_caption'],
            info['ground_truth_caption'],
            pred_caption
        )
        
        # 判断是否匹配
        is_caption_match = info['ground_truth_caption'].strip() == pred_caption.strip()
        
        result = {
            "index": info['idx'],
            "image_path": info['image_path'],
            "saved_image_path": info['saved_image_path'],
            
            # 三个 caption
            "original_caption": info['original_caption'],
            "ground_truth_caption": info['ground_truth_caption'],
            "predicted_caption": pred_caption,
            
            # Reasoning
            "ground_truth_reasoning": info['ground_truth_reasoning'],
            "predicted_reasoning": pred_reasoning,
            
            # 差异分析
            "diff_analysis": {
                "original_vs_ground_truth": diffs['original_vs_ground_truth'],
                "original_vs_predicted": diffs['original_vs_predicted'],
                "ground_truth_vs_predicted": diffs['ground_truth_vs_predicted'],
                "summary": diffs['analysis']
            },
            
            # 完整模型输出
            "model_raw_output": prediction,
            
            # 匹配标记
            "is_caption_match": is_caption_match,
            "success": "[ERROR]" not in prediction
        }
        results.append(result)
    
    return results


def run_batch_inference(
    data: List[Dict],
    engine: Qwen3VLvLLMEngine,
    output_dir: str,
    batch_size: int = 32,
    max_tokens: int = 1024,
    save_images: bool = True,
    checkpoint_interval: int = 10
) -> List[Dict]:
    """Run batch inference on all data with checkpointing."""
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    image_dir = os.path.join(output_dir, "images") if save_images else None
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    all_results = checkpoint["results"]
    start_idx = checkpoint["processed_count"]
    
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}")
    
    total_samples = len(data)
    num_batches = (total_samples - start_idx + batch_size - 1) // batch_size
    
    print(f"\nProcessing {total_samples - start_idx} remaining samples in {num_batches} batches...")
    print(f"Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}")
    
    # Process in batches
    for batch_num in tqdm(range(num_batches), desc="Batch inference"):
        batch_start = start_idx + batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        
        batch_data = data[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        # Process batch
        batch_results = process_batch(
            batch_data,
            batch_indices,
            engine,
            image_dir,
            max_tokens
        )
        
        all_results.extend(batch_results)
        
        # Save checkpoint periodically
        if (batch_num + 1) % checkpoint_interval == 0 or batch_end == total_samples:
            save_checkpoint(checkpoint_path, all_results, batch_end)
            print(f"\nCheckpoint saved: {batch_end}/{total_samples} samples processed")
    
    return all_results


# ================== Results Saving ==================

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics."""
    total = len(results)
    success = sum(1 for r in results if r.get('success', False))
    caption_matches = sum(1 for r in results if r.get('is_caption_match', False))
    errors = sum(1 for r in results if not r.get('success', False))
    
    metrics = {
        "total_samples": total,
        "successful_inferences": success,
        "inference_errors": errors,
        "success_rate": success / total if total > 0 else 0,
        "caption_exact_matches": caption_matches,
        "caption_match_rate": caption_matches / total if total > 0 else 0
    }
    
    return metrics


def save_results(results: List[Dict], metrics: Dict, output_dir: str, args):
    """Save final results in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存完整结果
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved full results to: {results_path}")
    
    # 2. 保存仅 caption 对比的简化结果
    caption_comparison = []
    for r in results:
        caption_comparison.append({
            "index": r["index"],
            "image_path": r.get("saved_image_path") or r.get("image_path"),
            "original_caption": r["original_caption"],
            "ground_truth_caption": r["ground_truth_caption"],
            "predicted_caption": r["predicted_caption"],
            "is_match": r["is_caption_match"],
            "diff_summary": r.get("diff_analysis", {}).get("summary", "")
        })
    
    comparison_path = os.path.join(output_dir, "caption_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(caption_comparison, f, ensure_ascii=False, indent=2)
    print(f"Saved caption comparison to: {comparison_path}")
    
    # 3. 保存差异分析详情
    diff_analysis_results = []
    for r in results:
        diff_info = r.get("diff_analysis", {})
        diff_analysis_results.append({
            "index": r["index"],
            "image_path": r.get("saved_image_path") or r.get("image_path"),
            "is_match": r["is_caption_match"],
            "summary": diff_info.get("summary", ""),
            
            # 原始 vs Ground Truth (期望的修正)
            "expected_correction": {
                "has_changes": not diff_info.get("original_vs_ground_truth", {}).get("is_identical", True),
                "similarity": diff_info.get("original_vs_ground_truth", {}).get("diff_ratio", 1.0),
                "changes": diff_info.get("original_vs_ground_truth", {}).get("changes", [])
            },
            
            # 原始 vs 预测 (模型实际修正)
            "model_correction": {
                "has_changes": not diff_info.get("original_vs_predicted", {}).get("is_identical", True),
                "similarity": diff_info.get("original_vs_predicted", {}).get("diff_ratio", 1.0),
                "changes": diff_info.get("original_vs_predicted", {}).get("changes", [])
            },
            
            # Ground Truth vs 预测 (模型与标准答案的差异)
            "prediction_vs_ground_truth": {
                "is_identical": diff_info.get("ground_truth_vs_predicted", {}).get("is_identical", False),
                "similarity": diff_info.get("ground_truth_vs_predicted", {}).get("diff_ratio", 0.0),
                "changes": diff_info.get("ground_truth_vs_predicted", {}).get("changes", [])
            }
        })
    
    diff_path = os.path.join(output_dir, "diff_analysis.json")
    with open(diff_path, 'w', encoding='utf-8') as f:
        json.dump(diff_analysis_results, f, ensure_ascii=False, indent=2)
    print(f"Saved diff analysis to: {diff_path}")
    
    # 4. 保存不匹配的样本（用于分析）
    mismatches = [r for r in results if not r.get('is_caption_match', False) and r.get('success', False)]
    mismatches_path = os.path.join(output_dir, "mismatched_samples.json")
    with open(mismatches_path, 'w', encoding='utf-8') as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(mismatches)} mismatched samples to: {mismatches_path}")
    
    # 5. 保存 metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    
    # 6. 保存人类可读的差异报告
    diff_report_path = os.path.join(output_dir, "diff_report.txt")
    with open(diff_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Caption Difference Analysis Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("LEGEND:\n")
        f.write("  - Original vs GT: Expected corrections (what should be changed)\n")
        f.write("  - Original vs Pred: Model's actual corrections\n")
        f.write("  - GT vs Pred: Difference between expected and actual\n")
        f.write("\n")
        
        for r in results[:30]:  # 显示前30个样本
            diff_info = r.get("diff_analysis", {})
            
            f.write("=" * 80 + "\n")
            f.write(f"[Sample {r['index']}] {'✓ MATCH' if r['is_caption_match'] else '✗ MISMATCH'}\n")
            f.write(f"Image: {r.get('saved_image_path') or r.get('image_path')}\n")
            f.write(f"Summary: {diff_info.get('summary', 'N/A')}\n")
            f.write("-" * 80 + "\n\n")
            
            # Original vs Ground Truth
            orig_gt = diff_info.get("original_vs_ground_truth", {})
            f.write("【Original → Ground Truth】(Expected Correction)\n")
            if orig_gt.get("is_identical", True):
                f.write("  No changes expected (caption is correct)\n")
            else:
                f.write(f"  Similarity: {orig_gt.get('diff_ratio', 0):.2%}\n")
                for change in orig_gt.get("changes", [])[:5]:
                    if change['type'] == 'replace':
                        f.write(f"  ✎ REPLACE: \"{change['original']}\" → \"{change['new']}\"\n")
                    elif change['type'] == 'delete':
                        f.write(f"  ✖ DELETE: \"{change['original']}\"\n")
                    elif change['type'] == 'insert':
                        f.write(f"  ✚ INSERT: \"{change['new']}\"\n")
                if len(orig_gt.get("changes", [])) > 5:
                    f.write(f"  ... and {len(orig_gt['changes']) - 5} more changes\n")
            f.write("\n")
            
            # Original vs Predicted
            orig_pred = diff_info.get("original_vs_predicted", {})
            f.write("【Original → Predicted】(Model's Correction)\n")
            if orig_pred.get("is_identical", True):
                f.write("  No changes made by model\n")
            else:
                f.write(f"  Similarity: {orig_pred.get('diff_ratio', 0):.2%}\n")
                for change in orig_pred.get("changes", [])[:5]:
                    if change['type'] == 'replace':
                        f.write(f"  ✎ REPLACE: \"{change['original']}\" → \"{change['new']}\"\n")
                    elif change['type'] == 'delete':
                        f.write(f"  ✖ DELETE: \"{change['original']}\"\n")
                    elif change['type'] == 'insert':
                        f.write(f"  ✚ INSERT: \"{change['new']}\"\n")
                if len(orig_pred.get("changes", [])) > 5:
                    f.write(f"  ... and {len(orig_pred['changes']) - 5} more changes\n")
            f.write("\n")
            
            # Ground Truth vs Predicted
            gt_pred = diff_info.get("ground_truth_vs_predicted", {})
            f.write("【Ground Truth vs Predicted】(Accuracy Check)\n")
            if gt_pred.get("is_identical", False):
                f.write("  ✓ PERFECT MATCH!\n")
            else:
                f.write(f"  ✗ MISMATCH - Similarity: {gt_pred.get('diff_ratio', 0):.2%}\n")
                for change in gt_pred.get("changes", [])[:5]:
                    if change['type'] == 'replace':
                        f.write(f"  ✎ GT has \"{change['original']}\" but model produced \"{change['new']}\"\n")
                    elif change['type'] == 'delete':
                        f.write(f"  ✖ GT has \"{change['original']}\" but model removed it\n")
                    elif change['type'] == 'insert':
                        f.write(f"  ✚ Model added \"{change['new']}\" not in GT\n")
                if len(gt_pred.get("changes", [])) > 5:
                    f.write(f"  ... and {len(gt_pred['changes']) - 5} more differences\n")
            f.write("\n")
    
    print(f"Saved diff report to: {diff_report_path}")
    
    # 7. 保存评估报告
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Qwen3-VL Caption Correction Evaluation Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Input data: {args.input_data}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Tensor parallel: {args.tensor_parallel_size}\n")
        f.write("\n")
        
        f.write("METRICS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {metrics['total_samples']}\n")
        f.write(f"Successful inferences: {metrics['successful_inferences']}\n")
        f.write(f"Inference errors: {metrics['inference_errors']}\n")
        f.write(f"Success rate: {metrics['success_rate']:.2%}\n")
        f.write(f"Caption exact matches: {metrics['caption_exact_matches']}\n")
        f.write(f"Caption match rate: {metrics['caption_match_rate']:.2%}\n")
        f.write("\n")
        
        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"- inference_results.json: Full results with all fields\n")
        f.write(f"- caption_comparison.json: Simplified caption comparison\n")
        f.write(f"- diff_analysis.json: Detailed diff analysis\n")
        f.write(f"- diff_report.txt: Human-readable diff report\n")
        f.write(f"- mismatched_samples.json: Samples where prediction != ground truth\n")
        f.write(f"- metrics.json: Evaluation metrics\n")
    
    print(f"Saved evaluation report to: {report_path}")


def print_summary(metrics: Dict):
    """Print summary."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total samples:        {metrics['total_samples']}")
    print(f"Successful:           {metrics['successful_inferences']}")
    print(f"Errors:               {metrics['inference_errors']}")
    print(f"Success rate:         {metrics['success_rate']:.2%}")
    print("-" * 60)
    print(f"Caption matches:      {metrics['caption_exact_matches']}")
    print(f"Caption match rate:   {metrics['caption_match_rate']:.2%}")
    print("=" * 60)


# ================== CLI ==================

def main():
    parser = argparse.ArgumentParser(
        description='Batch offline inference for Qwen3-VL using vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch inference
  python evaluate_finetuned_model.py \\
      --input-data /path/to/eval_data.json \\
      --model-path /path/to/model \\
      --output-dir ./output \\
      --batch-size 32

  # Resume from checkpoint
  python evaluate_finetuned_model.py \\
      --input-data /path/to/eval_data.json \\
      --model-path /path/to/model \\
      --output-dir ./output \\
      --resume

  # Sample subset for testing
  python evaluate_finetuned_model.py \\
      --input-data /path/to/eval_data.json \\
      --model-path /path/to/model \\
      --output-dir ./output \\
      --num-samples 100
        """
    )
    
    parser.add_argument(
        '--input-data', '-i',
        type=str,
        default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_bbox_sharegpt_eval.json",
        help='Path to input dataset (JSON or JSONL)'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/models/Qwen3-VL-32B-SFT-bbox_no_reason_plot",
        help='Path to Qwen3-VL model'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/output/rewriter/eval/32B-7w-bbox-no-reason-plot',
        help='Output directory (default: ./inference_output)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=-1,
        help='Number of samples to process (-1 for all, default: -1)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    parser.add_argument(
        '--tensor-parallel-size', '-tp',
        type=int,
        default=8,
        help='Tensor parallel size (default: auto-detect)'
    )
    
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=32768,
        help='Max model context length (default: 32768)'
    )
    
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization (default: 0.9)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Max tokens to generate (default: 1024)'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Save checkpoint every N batches (default: 10)'
    )
    
    parser.add_argument(
        '--no-save-images',
        action='store_true',
        help='Do not save images'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if exists'
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_input_data(args.input_data)
    
    # Sample if needed
    if args.num_samples > 0:
        data = sample_data(data, args.num_samples, args.seed)
    
    # Clear checkpoint if not resuming
    if not args.resume:
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Cleared existing checkpoint")
    
    # Initialize engine
    engine = Qwen3VLvLLMEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    engine.load_model()
    
    # Run batch inference
    results = run_batch_inference(
        data=data,
        engine=engine,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        save_images=not args.no_save_images,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Save final results
    metrics = compute_metrics(results)
    save_results(results, metrics, args.output_dir, args)
    print_summary(metrics)
    
    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
