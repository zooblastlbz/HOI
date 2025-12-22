import os
import json
import argparse
import time
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    Modality,
    Blob,
    Content,
    Part,
)

# ================== 基础配置 ==================

SEED = 42
np.random.seed(SEED)

# 替换成你自己的凭证路径
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    '/pfs/gaohuan03/gemini_exp/mmu-gemini-caption-1-5pro-86ec97219196.json'
)
user_info_path = '/pfs/gaohuan03/gemini_exp/mmu-gemini-caption-1-5pro-86ec97219196.json'

try:
    user_info = json.load(open(user_info_path))
    PROJECT_ID = user_info['project_id']
except:
    PROJECT_ID = "your-project-id"

LOCATION = 'global'

# 建议使用支持 Thinking 的模型以获得最佳效果
MODEL = 'gemini-3-pro-preview' 

# 保持宽松的安全设置
safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
]

try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"Warning: Client init failed: {e}")

# 并发写入锁
_write_lock = Lock()


def _safe_write_output(path: str, results: List[Optional[Dict]]):
    """
    线程安全地写出当前阶段的结果 JSON。
    """
    valid_items = [r for r in results if r is not None]
    
    total_processed = len(valid_items)
    total_success = sum(1 for r in valid_items if r.get('success', False))
    total_modified = sum(1 for r in valid_items if r.get('has_modification', False) and r.get('success', False))

    payload = {
        'meta': {
            'total_items': len(results),
            'processed_count': total_processed,
            'success_count': total_success,
            'modified_count': total_modified
        },
        'results': results, 
    }

    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"Error writing output file: {e}")


# ================== 图像绘制工具 ==================

def visualize_pose(image_bytes: bytes, pose_data: Dict) -> bytes:
    """在图像上绘制 Pose 关键点和标签。"""
    if not pose_data or not pose_data.get('success'):
        return image_bytes

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    
    h, w, _ = img.shape
    font_scale = max(0.5, w / 1500.0)
    thickness = max(1, int(w / 800.0))
    circle_radius = max(3, int(w / 300.0))

    persons = pose_data.get('persons', [])
    for person in persons:
        keypoints = person.get('keypoints', {})
        if not keypoints: continue
            
        for kp_name, kp_info in keypoints.items():
            if kp_info is None: continue
            
            x, y = int(kp_info['x']), int(kp_info['y'])
            
            # 颜色编码仅作为辅助参考
            if 'left' in kp_name:
                color = (0, 0, 255) # Red for LEFT anatomical side
                short_name = "L_" + kp_name.replace("left_", "")
            elif 'right' in kp_name:
                color = (255, 0, 0) # Blue for RIGHT anatomical side
                short_name = "R_" + kp_name.replace("right_", "")
            else:
                color = (0, 255, 0) 
                short_name = kp_name
            
            # =========================================================
            # 修改点：将 -1 改为 thickness，绘制空心圆
            # =========================================================
            cv2.circle(img, (x, y), circle_radius, color, thickness)
            
            text_pos = (x + 10, y - 10)
            cv2.putText(img, short_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, short_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()

# ================== 调用 Gemini 做 Caption 修正 ==================

def correct_caption_with_gemini(image_path: str, original_caption: str, pose_annotation: Dict) -> Dict[str, Any]:
    """
    发送图片和 Pose 给 Gemini。
    """
    try:
        with open(image_path, 'rb') as f:
            raw_image_bytes = f.read()
        visualized_image_bytes = visualize_pose(raw_image_bytes, pose_annotation)
    except Exception as e:
        return {
            "success": False,
            "corrected_caption": original_caption,
            "modifications": [f"IMAGE_LOAD_ERROR: {str(e)}"],
            "reasoning": "",
            "has_modification": False
        }

    pose_str = json.dumps(pose_annotation, indent=2)

    # =========================================================================
    # System Prompt 更新
    # 核心：使用 "The Person's Own Side" 概念，强调详尽推理
    # =========================================================================
# =========================================================================
    # System Prompt Modified
    # 修改逻辑：
    # 1. Exhaustive Verification: 检查每一个提及的部位。
    # 2. Case A (Correction): 只要写错了左右，必须改。
    # 3. Case B (Conditional Clarification): 没写左右时，仅当"必要区分"时才加。
    # =========================================================================
    system_prompt = (
        "You are an expert image caption editor specializing in anatomical spatial accuracy (left/right consistency). "
        "Your goal is to ensure descriptions of body parts match the image perfectly.\n\n"
        
        "**CRITICAL CONCEPT: Perspective Alignment**\n"
        "You must distinguish between 'Viewer's Side' (screen) and 'The Person's Own Side' (subject) to ensure accuracy.\n"
        "- **Facing Camera:** Subject's Right hand is on Viewer's Left.\n"
        "- **Facing Away:** Subject's Right hand is on Viewer's Right.\n\n"
        
        "**Input Context:**\n"
        "You will receive a caption and potentially auxiliary spatial tags. "
        "**WARNING:** Spatial tags are NOT ground truth. Rely on visual evidence from the image first.\n\n"
        
        "**Rules:**\n"
        "1. **Exhaustive Verification:** You must identify and verify **EVERY** single body part mentioned in the original caption. Do not skip any.\n"
        "2. **Strict Scope:** Focus **ONLY** on human body parts already mentioned in the text. DO NOT change descriptions of objects, backgrounds, or sentence styles.\n"
        "3. **Modification Criteria (Apply to all mentioned parts):** \n"
        "   - **Case A: Error Correction:** If the caption explicitly states a side (left/right) that is visually INCORRECT (based on the Person's Own Side), you must CORRECT it.\n"
        "   - **Case B: Conditional Clarification:** If a limb is mentioned WITHOUT a side (e.g., 'hand'), ONLY add 'left' or 'right' if it is **necessary to distinguish** between the two limbs (e.g., they are performing different actions or have different features). **DO NOT** add the side if the action is generic and the specific side is irrelevant to the clarity of the scene.\n"
        "4. **Preservation:** Keep the original sentence structure and vocabulary exactly as is for all non-body-part content.\n\n"
        
        "Return the output STRICTLY in JSON format with the following keys:\n"
        "- 'reasoning': A detailed analysis string explaining the Left/Right logic. Explicitly state why a clarification was added (because of distinct actions) or skipped (because it was unnecessary).\n"
        "- 'corrected_caption': The full caption string (corrected or original).\n"
        "- 'modifications': A list of strings describing the changes. **Each string must include the surrounding context**. Format: '... context words [original -> corrected] context words ...'. Example: 'holding a cup in his [hand -> left hand]'. Return [] if no changes.\n"
    )

    user_prompt = (
        f"Original Caption: \"{original_caption}\"\n\n"
        f"Pose Annotation Data (Reference): \n```json\n{pose_str}\n```"
    )

    contents = [
        Content(
            role="user",
            parts=[
                Part(text=system_prompt + "\n" + user_prompt),
                Part(inline_data=Blob(mime_type="image/jpeg", data=visualized_image_bytes)),
            ],
        )
    ]

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=GenerateContentConfig(
                safety_settings=safety_settings,
                temperature=1.0,
                top_p=1.0, 
                seed=SEED,
                max_output_tokens=65536,
                response_mime_type="application/json",
                # 依然启用 Thinking 让模型更聪明，但不提取其原始输出
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                ),
            ),
        )

        json_text = ''

        # 解析 Response，忽略 Thinking Part
        if resp.candidates:
            for c in resp.candidates:
                if c.finish_reason and c.finish_reason.name == "MAX_TOKENS":
                    print(f"Warning: Response truncated (MAX_TOKENS) for image {os.path.basename(image_path)}")

                for p in c.content.parts:
                    # 仅提取非 thought 的文本部分 (即 JSON)
                    if not getattr(p, 'thought', False):
                        json_text += (p.text if p.text else "")
        
        json_text = json_text.strip()
        if json_text.startswith("```json"): json_text = json_text[7:]
        if json_text.startswith("```"): json_text = json_text[3:] 
        if json_text.endswith("```"): json_text = json_text[:-3]
        
        result_json = json.loads(json_text)
        
        corrected = result_json.get("corrected_caption", original_caption)
        mods = result_json.get("modifications", [])
        reasoning = result_json.get("reasoning", "")
        
        return {
            "success": True,
            "corrected_caption": corrected,
            "modifications": mods,
            "reasoning": reasoning, 
            "has_modification": len(mods) > 0 and (corrected != original_caption)
            # Removed 'thinking_summary'
        }

    except Exception as e:
        return {
            "success": False,
            "corrected_caption": original_caption,
            "modifications": [f"API_OR_PARSE_ERROR: {str(e)}"],
            "reasoning": "",
            "has_modification": False
        }


# ================== 单条样本处理 ==================

def _process_single(idx: int, item: Dict) -> Dict:
    image_path = item.get('image_path')
    original_caption = item.get('caption', '')
    pose_annotation = item.get('pose_annotation', {})

    if not image_path or not os.path.exists(image_path):
         return {
            'idx': idx,
            'image_path': image_path,
            'original_caption': original_caption,
            'corrected_caption': original_caption,
            'modifications': ["FILE_NOT_FOUND"],
            'reasoning': "File not found",
            'has_modification': False,
            'success': False
        }

    gemini_res = correct_caption_with_gemini(image_path, original_caption, pose_annotation)

    return {
        'idx': idx,
        'image_path': image_path,
        'original_caption': original_caption,
        'corrected_caption': gemini_res['corrected_caption'],
        'modifications': gemini_res['modifications'],
        'reasoning': gemini_res['reasoning'], 
        'has_modification': gemini_res['has_modification'],
        'success': gemini_res['success']
    }


# ================== 主流程 ==================

def process_captions(
    input_json_path: str,
    output_json_path: Optional[str] = None,
    max_workers: int = 8,
):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data: List[Dict] = json.load(f)

    total = len(data)
    if total == 0:
        print('No samples.')
        return

    finished_results: List[Optional[Dict]] = [None] * total
    
    # 断点续跑逻辑
    if output_json_path is not None and os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            
            prev_results_list = prev.get('results', [])
            
            loaded_count = 0
            success_count = 0
            
            for item in prev_results_list:
                if item is None: continue
                idx = item.get('idx')
                if idx is not None and 0 <= idx < total:
                    finished_results[idx] = item
                    loaded_count += 1
                    if item.get('success', False):
                        success_count += 1
            
            print(f"Resuming from {output_json_path}")
            print(f"Loaded {loaded_count} previous records.")
            print(f"  - Success: {success_count}, To Retry: {loaded_count - success_count}")
            
        except Exception as e:
            print(f"Failed to load existing output (starting fresh): {e}")

    remaining_indices = []
    for i in range(total):
        res = finished_results[i]
        if res is None or not res.get('success', False):
            remaining_indices.append(i)

    remaining_count = len(remaining_indices)

    if remaining_count == 0:
        print('All samples successfully processed.')
        return

    print(f'Starting processing. Tasks: {remaining_count}')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single, idx, data[idx]): idx
            for idx in remaining_indices
        }
        
        pbar = tqdm(as_completed(future_to_idx), total=remaining_count, desc='Gemini Processing')
        
        for future in pbar:
            res = future.result()
            idx = res['idx']
            finished_results[idx] = res

            is_success = res['success']
            is_mod = res['has_modification']
            has_reason = len(res.get('reasoning', '')) > 0
            
            # 进度条不再显示 Think
            pbar.set_postfix({
                'idx': idx, 
                'OK': is_success, 
                'Mod': is_mod, 
                'Rsn': has_reason
            })

            if output_json_path is not None:
                with _write_lock:
                    _safe_write_output(output_json_path, finished_results)

    if output_json_path is not None:
        with _write_lock:
            _safe_write_output(output_json_path, finished_results)
            
    print(f"Done. Results saved to {output_json_path}")


# ================== CLI ==================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gemini Caption Refinement with Pose Reference.')
    
    parser.add_argument('--input_json', type=str, 
                        default='/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_dataset.json',
                        help='Input JSON')
    
    parser.add_argument('--output_json', type=str, 
                        default='/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_dataset.json',
                        help='Output JSON path')
    
    parser.add_argument('--num_threads', type=int, default=100, 
                        help='Concurrency level')
    
    
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("Error: 'opencv-python' is not installed.")
        exit(1)

    process_captions(args.input_json, args.output_json, max_workers=args.num_threads)