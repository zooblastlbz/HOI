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
        
        annotations = item.get('uncertain_parts', [])
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
        
        item['uncertain_parts'] = updated_annotations
    
    print(f"Laterality decision complete:")
    print(f"  - Total parts: {total_parts}")
    print(f"  - Valid laterality: {valid_parts}")
    print(f"  - Failed by count filter: {count_filter_failed}")
    
    return items


# 可以独立运行测试
if __name__ == "__main__":
    # 测试用例
    test_part = {
        'grounding_valid': True,
        'body_part': 'hand',
        'body_part_bbox': [150, 100, 200, 150],
        'body_part_count': 1,
        'left_body_part_bbox': [140, 95, 190, 145],
        'left_body_part_count': 1,
        'right_body_part_bbox': [250, 100, 300, 150],
        'right_body_part_count': 1,
        'interaction_object': 'coffee cup',
        'object_bboxes': [[130, 110, 160, 140]]
    }
    
    result = decide_laterality(test_part)
    print("Test result:")
    print(f"  final_laterality: {result.get('final_laterality')}")
    print(f"  laterality_decision_method: {result.get('laterality_decision_method')}")
    print(f"  laterality_valid: {result.get('laterality_valid')}")
    print(f"  final_body_part_bbox: {result.get('final_body_part_bbox')}")
    
    # 测试数量筛选失败的情况
    test_part_invalid = {
        'grounding_valid': True,
        'body_part': 'hand',
        'body_part_bbox': [150, 100, 200, 150],
        'body_part_count': 2,  # 无效：检测到2个
        'left_body_part_bbox': [140, 95, 190, 145],
        'left_body_part_count': 1,
        'right_body_part_bbox': [250, 100, 300, 150],
        'right_body_part_count': 1,
    }
    
    result_invalid = decide_laterality(test_part_invalid)
    print("\nTest invalid count result:")
    print(f"  laterality_valid: {result_invalid.get('laterality_valid')}")
    print(f"  laterality_error: {result_invalid.get('laterality_error')}")
