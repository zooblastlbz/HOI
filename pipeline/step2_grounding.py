import os
import torch
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("Warning: transformers library not found or Sam3 not available.")
    Sam3Processor = None
    Sam3Model = None

"""
Step 2.5: Laterality Decision Module
根据 grounding 结果判定身体部位的左右侧
"""

import math
from typing import List, Dict, Optional, Tuple


def compute_bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """
    计算 bounding box 的中心点
    bbox: [x1, y1, x2, y2]
    返回: (center_x, center_y)
    """
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    计算两点之间的欧氏距离
    """
    if point1 is None or point2 is None:
        return float('inf')
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_min_distance_to_objects(body_part_bbox: List[int], object_bboxes: List[List[int]]) -> float:
    """
    计算身体部位 bbox 与多个物体 bbox 的最小距离
    """
    if body_part_bbox is None or not object_bboxes:
        return float('inf')
    
    body_center = compute_bbox_center(body_part_bbox)
    if body_center is None:
        return float('inf')
    
    min_dist = float('inf')
    for obj_bbox in object_bboxes:
        if obj_bbox is None:
            continue
        obj_center = compute_bbox_center(obj_bbox)
        if obj_center is None:
            continue
        dist = compute_distance(body_center, obj_center)
        min_dist = min(min_dist, dist)
    
    return min_dist


def validate_bbox_counts(part: Dict) -> Tuple[bool, str]:
    """
    验证 grounding 结果的 bbox 数量是否符合要求
    
    筛选条件：
    - body_part_count: 动作描述 grounding 结果数量必须为 1
    - left_body_part_count: 左侧 grounding 结果数量必须为 1
    - right_body_part_count: 右侧 grounding 结果数量必须为 1
    
    返回: (is_valid, error_message)
    """
    # 检查动作描述 grounding 数量
    action_count = part.get('body_part_count', 1)
    if action_count != 1:
        return False, f"action_grounding_count_invalid: expected 1, got {action_count}"
    
    # 检查左侧 grounding 数量
    left_count = part.get('left_body_part_count', 1)
    if left_count != 1:
        return False, f"left_grounding_count_invalid: expected 1, got {left_count}"
    
    # 检查右侧 grounding 数量
    right_count = part.get('right_body_part_count', 1)
    if right_count != 1:
        return False, f"right_grounding_count_invalid: expected 1, got {right_count}"
    
    return True, ""


def decide_laterality_by_action(part: Dict) -> Optional[str]:
    """
    仅根据动作描述的 grounding 结果判定左右
    
    逻辑：比较 left_body_part_bbox 和 right_body_part_bbox 与 body_part_bbox 的距离，
    距离更近的那个就是正确的左右判定
    
    返回: "left", "right", 或 None（无法判定）
    """
    body_part_bbox = part.get('body_part_bbox')
    left_bbox = part.get('left_body_part_bbox')
    right_bbox = part.get('right_body_part_bbox')
    
    if body_part_bbox is None:
        return None
    
    if left_bbox is None and right_bbox is None:
        return None
    
    body_center = compute_bbox_center(body_part_bbox)
    left_center = compute_bbox_center(left_bbox)
    right_center = compute_bbox_center(right_bbox)
    
    left_dist = compute_distance(body_center, left_center)
    right_dist = compute_distance(body_center, right_center)
    
    if left_dist < right_dist:
        return "left"
    elif right_dist < left_dist:
        return "right"
    else:
        return None  # 距离相等，无法判定


def decide_laterality_by_object(part: Dict) -> Optional[str]:
    """
    根据交互对象的 grounding 结果判定左右
    
    逻辑：计算 left_body_part_bbox 和 right_body_part_bbox 与所有交互对象 bbox 的最小距离，
    最小距离更小的那个就是正确的左右判定
    
    返回: "left", "right", 或 None（无法判定）
    """
    object_bboxes = part.get('object_bboxes', [])
    left_bbox = part.get('left_body_part_bbox')
    right_bbox = part.get('right_body_part_bbox')
    
    if not object_bboxes:
        return None
    
    if left_bbox is None and right_bbox is None:
        return None
    
    left_min_dist = compute_min_distance_to_objects(left_bbox, object_bboxes)
    right_min_dist = compute_min_distance_to_objects(right_bbox, object_bboxes)
    
    if left_min_dist < right_min_dist:
        return "left"
    elif right_min_dist < left_min_dist:
        return "right"
    else:
        return None  # 距离相等，无法判定


def decide_laterality(part: Dict) -> Dict:
    """
    根据 grounding 结果进行最终的左右判定
    
    判定逻辑：
    1. 首先验证 bbox 数量是否符合要求（动作描述、左侧、右侧各只能有一个）
    2. 如果只有动作描述（无交互对象）：根据 left/right bbox 与 body_part_bbox 的距离判定
    3. 如果有交互对象：根据 left/right bbox 与交互对象的最小距离判定
    4. 如果两者都存在：两者结果必须一致，否则标记为无效
    
    返回更新后的 part，增加以下字段：
    - final_laterality: "left" 或 "right" 或 None
    - final_body_part_bbox: 最终选定的 bbox（左或右）
    - laterality_decision_method: 判定方法
    - laterality_valid: 判定是否有效
    - laterality_error: 错误原因（如有）
    """
    result = part.copy()
    
    # 检查基础有效性
    if not part.get('grounding_valid', False):
        result['laterality_valid'] = False
        result['laterality_error'] = "grounding_invalid"
        result['final_laterality'] = None
        result['final_body_part_bbox'] = None
        return result
    
    # 验证 bbox 数量
    count_valid, count_error = validate_bbox_counts(part)
    if not count_valid:
        result['laterality_valid'] = False
        result['laterality_error'] = count_error
        result['final_laterality'] = None
        result['final_body_part_bbox'] = None
        return result
    
    body_part_bbox = part.get('body_part_bbox')
    left_bbox = part.get('left_body_part_bbox')
    right_bbox = part.get('right_body_part_bbox')
    interaction_object = part.get('interaction_object')
    object_bboxes = part.get('object_bboxes', [])
    
    # 如果 left 或 right 检测失败（为 None），则无法判定
    if left_bbox is None and right_bbox is None:
        result['laterality_valid'] = False
        result['laterality_error'] = "both_left_right_detection_failed"
        result['final_laterality'] = None
        result['final_body_part_bbox'] = None
        return result
    
    has_action = body_part_bbox is not None
    has_object = interaction_object is not None and len(object_bboxes) > 0
    
    action_decision = None
    object_decision = None
    
    # 根据动作描述判定
    if has_action:
        action_decision = decide_laterality_by_action(part)
    
    # 根据交互对象判定
    if has_object:
        object_decision = decide_laterality_by_object(part)
    
    # 综合判定
    if has_action and has_object:
        # 两者都存在，必须一致
        if action_decision is not None and object_decision is not None:
            if action_decision == object_decision:
                result['final_laterality'] = action_decision
                result['laterality_decision_method'] = "action_and_object_consistent"
                result['laterality_valid'] = True
            else:
                # 不一致，标记为无效
                result['final_laterality'] = None
                result['laterality_decision_method'] = "action_and_object_inconsistent"
                result['laterality_valid'] = False
                result['laterality_error'] = f"inconsistent: action={action_decision}, object={object_decision}"
        elif action_decision is not None:
            # 只有动作判定成功
            result['final_laterality'] = action_decision
            result['laterality_decision_method'] = "action_only"
            result['laterality_valid'] = True
        elif object_decision is not None:
            # 只有物体判定成功
            result['final_laterality'] = object_decision
            result['laterality_decision_method'] = "object_only"
            result['laterality_valid'] = True
        else:
            # 两者都无法判定
            result['final_laterality'] = None
            result['laterality_decision_method'] = "both_failed"
            result['laterality_valid'] = False
            result['laterality_error'] = "both_action_and_object_decision_failed"
    
    elif has_action:
        # 只有动作描述
        if action_decision is not None:
            result['final_laterality'] = action_decision
            result['laterality_decision_method'] = "action_only"
            result['laterality_valid'] = True
        else:
            result['final_laterality'] = None
            result['laterality_decision_method'] = "action_failed"
            result['laterality_valid'] = False
            result['laterality_error'] = "action_decision_failed"
    
    elif has_object:
        # 只有交互对象
        if object_decision is not None:
            result['final_laterality'] = object_decision
            result['laterality_decision_method'] = "object_only"
            result['laterality_valid'] = True
        else:
            result['final_laterality'] = None
            result['laterality_decision_method'] = "object_failed"
            result['laterality_valid'] = False
            result['laterality_error'] = "object_decision_failed"
    
    else:
        # 既没有动作也没有物体（理论上不应该发生）
        result['final_laterality'] = None
        result['laterality_decision_method'] = "no_reference"
        result['laterality_valid'] = False
        result['laterality_error'] = "no_action_or_object_for_decision"
    
    # 设置最终的 body part bbox
    if result.get('final_laterality') == "left":
        result['final_body_part_bbox'] = left_bbox
    elif result.get('final_laterality') == "right":
        result['final_body_part_bbox'] = right_bbox
    else:
        result['final_body_part_bbox'] = None
    
    return result


def process_laterality_decision(items: List[Dict]) -> List[Dict]:
    """
    批量处理左右判定
    
    Args:
        items: 包含 grounding 结果的 items 列表
        
    Returns:
        更新后的 items 列表，每个 annotation 增加左右判定结果
    """
    print("Processing laterality decision...")
    
    total_parts = 0
    valid_parts = 0
    count_filter_failed = 0
    
    for item in items:
        if item.get('grounding_skipped', False):
            continue
        
        annotations = item.get('"uncertain_parts"', [])
        if isinstance(annotations, str) or not annotations:
            continue
        
        updated_annotations = []
        for part in annotations:
            total_parts += 1
            updated_part = decide_laterality(part)
            if updated_part.get('laterality_valid', False):
                valid_parts += 1
            elif 'count_invalid' in updated_part.get('laterality_error', ''):
                count_filter_failed += 1
            updated_annotations.append(updated_part)
        
        item['"uncertain_parts"'] = updated_annotations
    
    print(f"Laterality decision complete:")
    print(f"  - Total parts: {total_parts}")
    print(f"  - Valid laterality: {valid_parts}")
    print(f"  - Failed by count filter: {count_filter_failed}")
    
    return items




class GroundingModule:
    def __init__(self, model_name="facebook/sam3-demo", gpu_ids=None, batch_size=1): 
        # 用户需指定正确的模型路径或 HuggingFace ID
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        self.num_gpus = len(self.gpu_ids)
        self.batch_size = batch_size
        self.device = f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        if Sam3Model:
            print(f"Loading SAM3 model: {model_name} on {self.num_gpus} GPU(s): {self.gpu_ids}, batch_size={self.batch_size}...")
            try:
                self.processor = Sam3Processor.from_pretrained(model_name)
                self.model = Sam3Model.from_pretrained(model_name)
                
                # 多GPU支持：使用 DataParallel
                if self.num_gpus > 1 and torch.cuda.is_available():
                    self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
                    self.model = self.model.to(self.device)
                else:
                    self.model = self.model.to(self.device)
                    
            except Exception as e:
                print(f"Failed to load SAM3 model: {e}")
                print("Ensure you have the correct model name and transformers version.")
        else:
            print("SAM3 modules not available in transformers.")

    def _convert_bbox(self, bbox):
        """
        Convert bbox to [x1, y1, x2, y2] format (list of ints).
        """
        if bbox is None:
            return None
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        return [int(b) for b in bbox]

    def _crop_image_by_bbox(self, image, bbox):
        """
        Crop image by bbox [x1, y1, x2, y2].
        Returns: cropped PIL Image and offset (x1, y1)
        """
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        return cropped, (x1, y1)

    def _adjust_bbox_to_original(self, bbox, offset):
        """
        Adjust bbox from cropped image coordinates back to original image coordinates.
        bbox: [x1, y1, x2, y2] in cropped image
        offset: (offset_x, offset_y) - the top-left corner of the crop in original image
        Returns: [x1, y1, x2, y2] in original image coordinates
        """
        if bbox is None:
            return None
        offset_x, offset_y = offset
        return [
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2] + offset_x,
            bbox[3] + offset_y
        ]

    def detect_person(self, image_path, person_description):
        """
        Detect the specific person described by the text using SAM3.
        Uses person_brief_description for grounding.
        Returns: 
            - Bounding Box [x1, y1, x2, y2] if exactly ONE person is detected
            - None if no person or multiple persons detected (invalid data)
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform person detection.")

        print(f"SAM3 processing person: '{person_description}' in {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            # SAM3 inference
            inputs = self.processor(images=[image], text=[person_description], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            # Results contain: masks, boxes, scores
            boxes = results[0].get('boxes', [])
            
            # 验证：人物检测必须只有1个结果
            if len(boxes) == 0:
                print(f"No person found for description: {person_description}")
                return None
            elif len(boxes) > 1:
                print(f"Multiple persons ({len(boxes)}) found for description: '{person_description}' - marking as invalid")
                return None  # 返回 None 表示数据无效
            
            # 只有1个人，直接使用返回的 box
            bbox = self._convert_bbox(boxes[0])
            return bbox

        except Exception as e:
            print(f"Error in detect_person: {e}")
            return None

    def detect_person_batch(self, items):
        """
        Batch detection for persons using person_brief_description.
        items: List of dicts with 'image_path' and 'person_brief_description'
        Returns: List of bounding boxes (None for invalid items with 0 or >1 persons)
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform batch detection.")

        results_list = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            images = []
            texts = []
            
            for item in batch:
                try:
                    image = Image.open(item['image_path']).convert("RGB")
                    images.append(image)
                    texts.append(item['person_brief_description'])
                except Exception as e:
                    print(f"Error loading image {item.get('image_path')}: {e}")
                    images.append(None)
                    texts.append("")
            
            # Filter valid items
            valid_indices = [j for j, img in enumerate(images) if img is not None]
            valid_images = [images[j] for j in valid_indices]
            valid_texts = [texts[j] for j in valid_indices]
            
            batch_results = [None] * len(batch)
            
            if valid_images:
                try:
                    inputs = self.processor(images=valid_images, text=valid_texts, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    post_results = self.processor.post_process_instance_segmentation(
                        outputs,
                        threshold=0.5,
                        mask_threshold=0.5,
                        target_sizes=inputs.get("original_sizes").tolist()
                    )
                    
                    for k, idx in enumerate(valid_indices):
                        boxes = post_results[k].get('boxes', [])
                        
                        # 验证：人物检测必须只有1个结果
                        if len(boxes) == 0:
                            print(f"No person found for item {idx}")
                            batch_results[idx] = None
                        elif len(boxes) > 1:
                            print(f"Multiple persons ({len(boxes)}) found for item {idx} - marking as invalid")
                            batch_results[idx] = None
                        else:
                            # 只有1个人，直接使用返回的 box
                            batch_results[idx] = self._convert_bbox(boxes[0])
                            
                except Exception as e:
                    print(f"Error in batch person detection: {e}")
            
            results_list.extend(batch_results)
        
        return results_list

    def detect_object(self, image_path, object_name):
        """
        Detect the interaction object in the full image using SAM3.
        Uses interaction_object field for grounding.
        Returns: List of Bounding Boxes [[x1, y1, x2, y2], ...] for all detected objects.
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform object detection.")

        print(f"SAM3 processing object: '{object_name}' in full image")
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # SAM3 inference on full image
            inputs = self.processor(images=[image], text=[object_name], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            # Results contain: masks, boxes, scores
            boxes = results[0].get('boxes', [])
            
            if len(boxes) == 0:
                print(f"No object found for: {object_name}")
                return []
            
            # 物品可以有多个，返回所有检测到的 boxes
            result_boxes = []
            for box in boxes:
                bbox = self._convert_bbox(box)
                if bbox:
                    result_boxes.append(bbox)
            
            return result_boxes

        except Exception as e:
            print(f"Error in detect_object: {e}")
            return []

    def detect_object_batch(self, items):
        """
        Batch detection for objects in full image.
        items: List of dicts with 'image_path', 'interaction_object'
        Returns: List of lists of bounding boxes (multiple objects allowed per item)
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform batch object detection.")

        results_list = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            images = []
            texts = []
            
            for item in batch:
                try:
                    image = Image.open(item['image_path']).convert("RGB")
                    images.append(image)
                    texts.append(item['interaction_object'])
                except Exception as e:
                    print(f"Error loading image {item.get('image_path')}: {e}")
                    images.append(None)
                    texts.append("")
            
            # Filter valid items
            valid_indices = [j for j, img in enumerate(images) if img is not None]
            valid_images = [images[j] for j in valid_indices]
            valid_texts = [texts[j] for j in valid_indices]
            
            batch_results = [[] for _ in batch]
            
            if valid_images:
                try:
                    inputs = self.processor(images=valid_images, text=valid_texts, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    post_results = self.processor.post_process_instance_segmentation(
                        outputs,
                        threshold=0.5,
                        mask_threshold=0.5,
                        target_sizes=inputs.get("original_sizes").tolist()
                    )
                    
                    for k, idx in enumerate(valid_indices):
                        boxes = post_results[k].get('boxes', [])
                        
                        # 物品可以有多个，返回所有检测到的 boxes
                        result_boxes = []
                        for box in boxes:
                            bbox = self._convert_bbox(box)
                            if bbox:
                                result_boxes.append(bbox)
                        
                        batch_results[idx] = result_boxes
                            
                except Exception as e:
                    print(f"Error in batch object detection: {e}")
            
            results_list.extend(batch_results)
        
        return results_list

    def detect_body_part_in_person_roi(self, image_path, person_bbox, action_description):
        """
        Detect body part within a person's bounding box using action_description.
        
        Args:
            image_path: Path to the image
            person_bbox: [x1, y1, x2, y2] bounding box of the person
            action_description: Description of the body part action for grounding
            
        Returns:
            - Bounding Box [x1, y1, x2, y2] in original image coordinates
            - None if detection fails
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform body part detection.")

        print(f"SAM3 processing body part: '{action_description}' within person ROI")
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Crop image to person ROI
            cropped_image, offset = self._crop_image_by_bbox(image, person_bbox)
            
            # SAM3 inference on cropped image
            inputs = self.processor(images=[cropped_image], text=[action_description], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=[(cropped_image.height, cropped_image.width)]
            )
            
            boxes = results[0].get('boxes', [])
            
            if len(boxes) == 0:
                print(f"No body part found for: {action_description}")
                return None
            
            # Take the first (highest confidence) detection
            bbox_in_crop = self._convert_bbox(boxes[0])
            
            # Convert back to original image coordinates
            bbox_in_original = self._adjust_bbox_to_original(bbox_in_crop, offset)
            
            return bbox_in_original

        except Exception as e:
            print(f"Error in detect_body_part_in_person_roi: {e}")
            return None

    def detect_body_part_batch(self, items):
        """
        Batch detection for body parts within person ROIs.
        
        items: List of dicts with:
            - 'image_path': path to image
            - 'person_bbox': [x1, y1, x2, y2] person bounding box
            - 'action_description': description for body part grounding
            
        Returns: List of tuples (bbox, count) where:
            - bbox: bounding box in original image coordinates (None for failed detections)
            - count: number of detected bboxes
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform batch body part detection.")

        results_list = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            cropped_images = []
            texts = []
            offsets = []
            valid_indices = []
            
            for j, item in enumerate(batch):
                try:
                    image = Image.open(item['image_path']).convert("RGB")
                    person_bbox = item['person_bbox']
                    
                    if person_bbox is None:
                        cropped_images.append(None)
                        texts.append("")
                        offsets.append((0, 0))
                        continue
                    
                    # Crop to person ROI
                    cropped_image, offset = self._crop_image_by_bbox(image, person_bbox)
                    cropped_images.append(cropped_image)
                    texts.append(item['action_description'])
                    offsets.append(offset)
                    valid_indices.append(j)
                    
                except Exception as e:
                    print(f"Error processing image {item.get('image_path')}: {e}")
                    cropped_images.append(None)
                    texts.append("")
                    offsets.append((0, 0))
            
            # Filter valid items for batch processing
            valid_images = [cropped_images[j] for j in valid_indices]
            valid_texts = [texts[j] for j in valid_indices]
            valid_offsets = [offsets[j] for j in valid_indices]
            
            batch_results = [(None, 0)] * len(batch)
            
            if valid_images:
                try:
                    # Get target sizes for cropped images
                    target_sizes = [(img.height, img.width) for img in valid_images]
                    
                    inputs = self.processor(images=valid_images, text=valid_texts, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    post_results = self.processor.post_process_instance_segmentation(
                        outputs,
                        threshold=0.5,
                        mask_threshold=0.5,
                        target_sizes=target_sizes
                    )
                    
                    for k, idx in enumerate(valid_indices):
                        boxes = post_results[k].get('boxes', [])
                        count = len(boxes)
                        
                        if count == 0:
                            print(f"No body part found for item {idx}")
                            batch_results[idx] = (None, 0)
                        else:
                            # Take the first (highest confidence) detection
                            bbox_in_crop = self._convert_bbox(boxes[0])
                            # Convert back to original image coordinates
                            bbox_in_original = self._adjust_bbox_to_original(bbox_in_crop, valid_offsets[k])
                            batch_results[idx] = (bbox_in_original, count)
                            
                except Exception as e:
                    print(f"Error in batch body part detection: {e}")
            
            results_list.extend(batch_results)
        
        return results_list

    def detect_left_right_body_parts_batch(self, items):
        """
        Batch detection for left and right body parts within person ROIs.
        For each body_part, detect both "left {body_part}" and "right {body_part}".
        
        items: List of dicts with:
            - 'image_path': path to image
            - 'person_bbox': [x1, y1, x2, y2] person bounding box
            - 'body_part': normalized body part name (e.g., "hand", "foot")
            
        Returns: List of dicts with:
            - 'left_bbox': left body part bbox in original image coordinates
            - 'left_count': number of detected left bboxes
            - 'right_bbox': right body part bbox in original image coordinates  
            - 'right_count': number of detected right bboxes
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform batch body part detection.")

        # 为每个 item 生成 left 和 right 的检测任务
        all_tasks = []  # (original_idx, side, image, text, offset)
        task_info = []  # (original_idx, side)
        
        for idx, item in enumerate(items):
            image_path = item['image_path']
            person_bbox = item['person_bbox']
            body_part = item['body_part']
            
            if person_bbox is None:
                continue
                
            try:
                image = Image.open(image_path).convert("RGB")
                cropped_image, offset = self._crop_image_by_bbox(image, person_bbox)
                
                # 添加 left 和 right 两个任务
                for side in ['left', 'right']:
                    text = f"{side} {body_part}"
                    all_tasks.append({
                        'cropped_image': cropped_image,
                        'text': text,
                        'offset': offset
                    })
                    task_info.append((idx, side))
                    
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        # 初始化结果
        results = [{'left_bbox': None, 'left_count': 0, 'right_bbox': None, 'right_count': 0} for _ in items]
        
        if not all_tasks:
            return results
        
        # 批量处理
        for i in range(0, len(all_tasks), self.batch_size):
            batch_tasks = all_tasks[i:i + self.batch_size]
            batch_info = task_info[i:i + self.batch_size]
            
            images = [t['cropped_image'] for t in batch_tasks]
            texts = [t['text'] for t in batch_tasks]
            offsets = [t['offset'] for t in batch_tasks]
            
            try:
                target_sizes = [(img.height, img.width) for img in images]
                inputs = self.processor(images=images, text=texts, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                post_results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=target_sizes
                )
                
                for k, (orig_idx, side) in enumerate(batch_info):
                    boxes = post_results[k].get('boxes', [])
                    count = len(boxes)
                    
                    if count > 0:
                        bbox_in_crop = self._convert_bbox(boxes[0])
                        bbox_in_original = self._adjust_bbox_to_original(bbox_in_crop, offsets[k])
                        results[orig_idx][f'{side}_bbox'] = bbox_in_original
                        results[orig_idx][f'{side}_count'] = count
                    else:
                        results[orig_idx][f'{side}_bbox'] = None
                        results[orig_idx][f'{side}_count'] = 0
                        
            except Exception as e:
                print(f"Error in batch left/right body part detection: {e}")
                # 标记这批任务失败
                for orig_idx, side in batch_info:
                    if f'{side}_bbox' not in results[orig_idx] or results[orig_idx][f'{side}_bbox'] is None:
                        results[orig_idx][f'{side}_bbox'] = None
                        results[orig_idx][f'{side}_count'] = 0
        
        return results

    def process_items_batch(self, items):
        """
        批量处理完整的 grounding 流程。
        
        流程:
        1. 使用 person_brief_description 对人进行 grounding，得到 person_bbox
        2. 使用 interaction_object 在全图中检测交互物体，得到 object_bboxes
        3. 在 person_bbox 内，使用 action_description 检测身体部位，得到 body_part_bbox
        4. 在 person_bbox 内，使用 "left/right {body_part}" 检测左右身体部位，得到 left_body_part_bbox 和 right_body_part_bbox
        
        Args:
            items: 列表，每个 item 包含:
                - image_path: 图片路径
                - "uncertain_parts": Step1 的输出，可以是：
                    - 字符串: "no_human_in_picture" 或 "all_body_parts_already_specified"
                    - 列表: 每个元素包含:
                        - person_brief_description: 简短人物描述（用于 grounding）
                        - person_full_description: 完整人物描述
                        - interaction_object: 交互物体描述（可为 null）
                        - action_description: 身体部位动作描述（用于 grounding）
                        - body_part: 身体部位名称（如 "hand", "foot"）
                        - laterality_specified: 是否指定了左右
                        - original_laterality: 原始左右信息
                        - text_span: 原文片段
                        - interaction_type: 交互类型
                        - person_id: 人物ID
                
        Returns:
            处理后的 items 列表，每个 annotation 增加:
                - person_bbox: 人物边界框
                - object_bboxes: 物体边界框列表
                - object_bbox: 第一个物体边界框
                - body_part_bbox: 身体部位边界框（使用 action_description）
                - left_body_part_bbox: 左侧身体部位边界框
                - right_body_part_bbox: 右侧身体部位边界框
                - grounding_valid: 是否有效
        """
        # 第一步：过滤，分离需要处理和不需要处理的 items
        items_to_process = []  # (index, annotations)
        
        for i, item in enumerate(items):
            annotations = item.get("uncertain_parts", [])
            
            # 过滤：跳过不需要处理的情况
            # Step1 返回字符串表示特殊情况
            if isinstance(annotations, str):
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = annotations
                continue
            
            # 空列表表示没有需要处理的身体部位
            if not annotations or len(annotations) == 0:
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = "no_body_parts_to_process"
                continue
            
            image_path = item.get('image_path')
            if not image_path or not os.path.exists(image_path):
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = "image_not_found"
                continue
            
            # 有需要处理的身体部位
            item['grounding_skipped'] = False
            items_to_process.append((i, annotations))
        
        if not items_to_process:
            print("No items need grounding processing.")
            return items
        
        print(f"Grounding: {len(items_to_process)} items need processing (out of {len(items)} total)")
        
        # ========== Step 1：批量检测人物 (使用 person_brief_description) ==========
        person_tasks = []
        task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, annotations in items_to_process:
            image_path = items[item_idx].get('image_path')
            for part_idx, part in enumerate(annotations):
                if 'person_bbox' not in part:
                    # 使用 person_brief_description 进行人物检测
                    person_desc = part.get('person_brief_description', '')
                    if not person_desc:
                        # 如果没有 brief description，尝试从 full description 提取
                        person_desc = part.get('person_full_description', '')
                    
                    person_tasks.append({
                        'image_path': image_path,
                        'person_brief_description': person_desc
                    })
                    task_mapping.append((item_idx, part_idx))
        
        if person_tasks:
            print(f"Step 1: Batch detecting {len(person_tasks)} persons using person_brief_description...")
            person_results = self.detect_person_batch(person_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), bbox in zip(task_mapping, person_results):
                items[item_idx]['"uncertain_parts"'][part_idx]['person_bbox'] = bbox
        
        # ========== Step 2：批量检测交互物体 (使用 interaction_object) ==========
        object_tasks = []
        obj_task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, _ in items_to_process:
            image_path = items[item_idx].get('image_path')
            annotations = items[item_idx]["uncertain_parts"]
            
            for part_idx, part in enumerate(annotations):
                person_bbox = part.get('person_bbox')
                # 使用 interaction_object 字段
                interaction_obj = part.get('interaction_object')
                
                # 只有人物检测成功且有交互物品时才检测物品
                # interaction_object 可能为 null/None，表示无物体交互
                if person_bbox and interaction_obj and 'object_bboxes' not in part:
                    object_tasks.append({
                        'image_path': image_path,
                        'interaction_object': interaction_obj
                    })
                    obj_task_mapping.append((item_idx, part_idx))
        
        if object_tasks:
            print(f"Step 2: Batch detecting {len(object_tasks)} objects using interaction_object...")
            object_results = self.detect_object_batch(object_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), obj_bboxes in zip(obj_task_mapping, object_results):
                part = items[item_idx]["uncertain_parts"][part_idx]
                part['object_bboxes'] = obj_bboxes
                part['object_bbox'] = obj_bboxes[0] if obj_bboxes else None
        
        # ========== Step 3：批量检测身体部位 (在 person_bbox 内使用 action_description) ==========
        body_part_tasks = []
        bp_task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, _ in items_to_process:
            image_path = items[item_idx].get('image_path')
            annotations = items[item_idx]["uncertain_parts"]
            
            for part_idx, part in enumerate(annotations):
                person_bbox = part.get('person_bbox')
                # 使用 action_description 字段进行身体部位检测
                action_desc = part.get('action_description')
                
                # 只有人物检测成功且有动作描述时才检测身体部位
                if person_bbox and action_desc and 'body_part_bbox' not in part:
                    body_part_tasks.append({
                        'image_path': image_path,
                        'person_bbox': person_bbox,
                        'action_description': action_desc
                    })
                    bp_task_mapping.append((item_idx, part_idx))
        
        if body_part_tasks:
            print(f"Step 3: Batch detecting {len(body_part_tasks)} body parts using action_description within person ROI...")
            body_part_results = self.detect_body_part_batch(body_part_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), (bp_bbox, count) in zip(bp_task_mapping, body_part_results):
                items[item_idx]["uncertain_parts"][part_idx]['body_part_bbox'] = bp_bbox
                items[item_idx]["uncertain_parts"][part_idx]['body_part_count'] = count
        
        # ========== Step 4：批量检测左右身体部位 (在 person_bbox 内使用 "left/right {body_part}") ==========
        lr_body_part_tasks = []
        lr_task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, _ in items_to_process:
            image_path = items[item_idx].get('image_path')
            annotations = items[item_idx]["uncertain_parts"]
            
            for part_idx, part in enumerate(annotations):
                person_bbox = part.get('person_bbox')
                body_part = part.get('body_part')
                
                # 只有人物检测成功且有 body_part 时才检测左右身体部位
                if person_bbox and body_part and 'left_body_part_bbox' not in part:
                    lr_body_part_tasks.append({
                        'image_path': image_path,
                        'person_bbox': person_bbox,
                        'body_part': body_part
                    })
                    lr_task_mapping.append((item_idx, part_idx))
        
        if lr_body_part_tasks:
            print(f"Step 4: Batch detecting {len(lr_body_part_tasks)} left/right body parts within person ROI...")
            lr_results = self.detect_left_right_body_parts_batch(lr_body_part_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), lr_bboxes in zip(lr_task_mapping, lr_results):
                part = items[item_idx]["uncertain_parts"][part_idx]
                part['left_body_part_bbox'] = lr_bboxes.get('left_bbox')
                part['left_body_part_count'] = lr_bboxes.get('left_count')
                part['right_body_part_bbox'] = lr_bboxes.get('right_bbox')
                part['right_body_part_count'] = lr_bboxes.get('right_count')
        
        # ========== Step 5：标记 grounding 有效性 ==========
        for item_idx, _ in items_to_process:
            annotations = items[item_idx]["uncertain_parts"]
            for part in annotations:
                if part.get('person_bbox') is None:
                    part['grounding_valid'] = False
                    part['grounding_error'] = "person_detection_failed"
                elif part.get('body_part_bbox') is None:
                    part['grounding_valid'] = False
                    part['grounding_error'] = "body_part_detection_failed"
                else:
                    part['grounding_valid'] = True
                    # 物体检测失败不影响整体有效性，但记录状态
                    if part.get('interaction_object') and not part.get('object_bbox'):
                        part['object_detection_failed'] = True
                    # 左右身体部位检测状态
                    if part.get('left_body_part_bbox') is None:
                        part['left_body_part_detection_failed'] = True
                    if part.get('right_body_part_bbox') is None:
                        part['right_body_part_detection_failed'] = True
        
        # ========== Step 6：左右判定 ==========
        print("Step 5: Processing laterality decision...")
        items = process_laterality_decision(items)
        
        return items
