import os
import cv2
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PoseModule:
    def __init__(self, model_path="yolo11n-pose.pt", gpu_ids=None, batch_size=1):
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        self.num_gpus = len(self.gpu_ids)
        self.batch_size = batch_size
        self.models = []
        self.device = f"cuda:{self.gpu_ids[0]}" if self.gpu_ids else "cpu"
        
        if YOLO:
            print(f"Loading YOLO11-Pose model from {model_path} on {self.num_gpus} GPU(s): {self.gpu_ids}, batch_size={self.batch_size}...")
            # 为每个GPU加载一个模型实例
            for gpu_id in self.gpu_ids:
                model = YOLO(model_path)
                self.models.append(model)
            # 主模型用于单个推理
            self.model = self.models[0] if self.models else None
        else:
            print("Warning: 'ultralytics' package not installed. Pose estimation will fail.")
            self.model = None

    # YOLO-Pose Keypoint Mapping (COCO format)
    # 输出顺序: 鼻子, 左眼, 右眼, 左耳, 右耳, 左肩, 右肩, 左肘, 右肘, 左腕, 右腕, 左髋, 右髋, 左膝, 右膝, 左踝, 右踝
    KEYPOINT_NAMES = [
        "nose",           # 0 - 鼻子
        "left_eye",       # 1 - 左眼
        "right_eye",      # 2 - 右眼
        "left_ear",       # 3 - 左耳
        "right_ear",      # 4 - 右耳
        "left_shoulder",  # 5 - 左肩
        "right_shoulder", # 6 - 右肩
        "left_elbow",     # 7 - 左肘
        "right_elbow",    # 8 - 右肘
        "left_wrist",     # 9 - 左腕
        "right_wrist",    # 10 - 右腕
        "left_hip",       # 11 - 左髋
        "right_hip",      # 12 - 右髋
        "left_knee",      # 13 - 左膝
        "right_knee",     # 14 - 右膝
        "left_ankle",     # 15 - 左踝
        "right_ankle"     # 16 - 右踝
    ]
    
    # 身体部位到关键点索引的映射
    BODY_PART_TO_INDICES = {
        "nose": (0, None),           # 鼻子只有一个
        "eye": (1, 2),               # 左眼, 右眼
        "ear": (3, 4),               # 左耳, 右耳
        "shoulder": (5, 6),          # 左肩, 右肩
        "elbow": (7, 8),             # 左肘, 右肘
        "wrist": (9, 10),            # 左腕, 右腕
        "hip": (11, 12),             # 左髋, 右髋
        "knee": (13, 14),            # 左膝, 右膝
        "ankle": (15, 16),           # 左踝, 右踝
    }
    
    # 置信度阈值
    CONFIDENCE_THRESHOLD_EXTRACT = 0.3   # 提取关键点的最低阈值
    CONFIDENCE_THRESHOLD_VALID = 0.95    # 判断左右时要求的高置信度阈值

    def _extract_keypoints(self, result, offset_x=0, offset_y=0):
        """
        从 YOLO 结果中提取关键点，并转换为全局坐标。
        YOLO 输出格式: keypoints.data shape (N, 17, 3) 其中 3 = (x, y, confidence)
        """
        keypoints_dict = {}
        
        if result is None or result.keypoints is None:
            return keypoints_dict
        
        # 获取关键点数据: (N, 17, 3) - N个人, 17个关键点, (x, y, conf)
        kpts_data = result.keypoints.data.cpu().numpy()
        
        if len(kpts_data) == 0:
            return keypoints_dict
        
        # 取第一个检测到的人（ROI 内应该只有一个目标人物）
        person_kpts = kpts_data[0]  # shape: (17, 3)
        
        # 提取所有 17 个关键点
        for i, name in enumerate(self.KEYPOINT_NAMES):
            x, y, conf = person_kpts[i]
            
            # 检查置信度和有效性（使用较低阈值提取，后续判断时使用高阈值）
            if conf < self.CONFIDENCE_THRESHOLD_EXTRACT or (x == 0 and y == 0):
                keypoints_dict[name] = None
            else:
                # 转换为全局坐标
                keypoints_dict[name] = {
                    "x": float(x + offset_x),
                    "y": float(y + offset_y),
                    "confidence": float(conf)
                }
        
        return keypoints_dict

    def get_keypoints(self, image_path, roi_bbox):
        """
        在人物的 ROI 区域内检测关键点。
        Returns: Dictionary of keypoints
        """
        if not self.model:
            return {}

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return {}

        # 解析 ROI (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, roi_bbox)
        
        # 边界检查
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            print("Invalid ROI for pose estimation.")
            return {}

        # 裁剪 ROI 区域
        roi_img = img[y1:y2, x1:x2]

        # 运行推理
        results = self.model(roi_img, device=self.device, verbose=False)
        
        if results:
            return self._extract_keypoints(results[0], x1, y1)
        
        return {}

    def get_keypoints_batch(self, items):
        """
        批量检测关键点。
        items: List of dicts with 'image_path' and 'roi_bbox'
        Returns: List of keypoint dictionaries
        """
        if not self.model:
            return [{} for _ in items]

        results_list = []
       
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            roi_images = []
            offsets = []  # Store (x1, y1) for coordinate conversion
            valid_indices = []
            
            for j, item in enumerate(batch):
                try:
                    image_path = item.get('image_path')
                    roi_bbox = item.get('roi_bbox')
                    
                    if not image_path or not roi_bbox:
                        continue
                    
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    x1, y1, x2, y2 = map(int, roi_bbox)
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    roi_img = img[y1:y2, x1:x2]
                    roi_images.append(roi_img)
                    offsets.append((x1, y1))
                    valid_indices.append(j)
                    
                except Exception as e:
                    print(f"Error loading image {item.get('image_path')}: {e}")
            
            batch_results = [{} for _ in batch]
            
            if roi_images:
                try:
                    # 批量推理
                    results = self.model(roi_images, device=self.device, verbose=False)
                    
                    for k, valid_idx in enumerate(valid_indices):
                        if k < len(results):
                            x1, y1 = offsets[k]
                            batch_results[valid_idx] = self._extract_keypoints(results[k], x1, y1)
                        
                except Exception as e:
                    print(f"Error in batch pose estimation: {e}")
            
            results_list.extend(batch_results)
        
        return results_list

    def process_items_batch(self, items):
        """
        批量处理完整的 pose estimation 流程。
        
        Args:
            items: 列表，每个 item 包含:
                - image_path: 图片路径
                - uncertain_body_part_annotation: 不确定身体部位的列表
                - grounding_skipped: 是否跳过了 grounding
                
        Returns:
            处理后的 items 列表
        """
        # 第一步：过滤，分离需要处理和不需要处理的 items
        print(f"Batch processing {len(items)} items for pose estimation...")
        items_to_process = []  # (index, annotations)
        
        for i, item in enumerate(items):
            # 检查是否在 Step 2 被跳过
            if item.get('grounding_skipped', False):
                item['pose_skipped'] = True
                item['pose_skip_reason'] = f"grounding was skipped: {item.get('grounding_skip_reason', 'unknown')}"
                continue
            
            image_path = item.get('image_path')
            if not image_path or not os.path.exists(image_path):
                item['pose_skipped'] = True
                item['pose_skip_reason'] = "image_not_found"
                continue
            
            annotations = item.get('uncertain_body_part_annotation', [])
            
            # 检查 annotations 类型
            if isinstance(annotations, str) or not annotations:
                item['pose_skipped'] = True
                item['pose_skip_reason'] = "no_valid_annotations"
                continue
            
            # 检查是否有有效的 person_bbox
            has_valid_bbox = False
            for part in annotations:
                if part.get('grounding_valid', False) and part.get('person_bbox'):
                    has_valid_bbox = True
                    break
            
            if not has_valid_bbox:
                item['pose_skipped'] = True
                item['pose_skip_reason'] = "no_valid_person_bbox"
                continue
            
            item['pose_skipped'] = False
            items_to_process.append((i, annotations))
        
        if not items_to_process:
            print("No items need pose estimation processing.")
            return items
        
        print(f"Pose Estimation: {len(items_to_process)} items need processing (out of {len(items)} total)")
        
        # 第二步：收集所有需要检测关键点的任务
        pose_tasks = []
        task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, annotations in items_to_process:
            image_path = items[item_idx].get('image_path')
            for part_idx, part in enumerate(annotations):
                # 只处理 grounding 有效且有 person_bbox 的
                if part.get('grounding_valid', False) and part.get('person_bbox'):
                    pose_tasks.append({
                        'image_path': image_path,
                        'roi_bbox': part.get('person_bbox')
                    })
                    task_mapping.append((item_idx, part_idx))
        
        # 第三步：批量检测关键点
        if pose_tasks:
            print(f"Batch detecting keypoints for {len(pose_tasks)} persons...")
            keypoints_results = self.get_keypoints_batch(pose_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), keypoints in zip(task_mapping, keypoints_results):
                part = items[item_idx]['uncertain_body_part_annotation'][part_idx]
                part['keypoints'] = keypoints
        
        # 第四步：根据关键点确定左右
        for item_idx, annotations in items_to_process:
            for part in items[item_idx]['uncertain_body_part_annotation']:
                # 跳过 grounding 无效的数据
                if not part.get('grounding_valid', False):
                    part["resolved_side"] = "skipped"
                    part["resolved_reason"] = "grounding invalid"
                    continue
                
                keypoints = part.get('keypoints', {})
                obj_bboxes = part.get('object_bboxes', [])
                interaction_types = part.get("interaction_category", [])
                body_part = part.get("body_part", "")
                
                if not keypoints:
                    part["resolved_side"] = "undetermined"
                    part["resolved_reason"] = "no keypoints detected"
                    continue
                
                # 只处理 close_proximity 类型，directional_orientation 暂不处理
                if "close_proximity" in interaction_types:
                    if obj_bboxes:
                        result = self._resolve_side_by_min_distance(keypoints, obj_bboxes, body_part)
                        part["resolved_side"] = result["side"]
                        part["resolved_reason"] = result["reason"]
                        if "details" in result:
                            part["distance_details"] = result["details"]
                    else:
                        part["resolved_side"] = "undetermined"
                        part["resolved_reason"] = "no object bboxes for close_proximity"
                
                elif "directional_orientation" in interaction_types:
                    # 朝向类型暂不处理
                    part["resolved_side"] = "skipped"
                    part["resolved_reason"] = "directional_orientation not yet supported"
                
                else:
                    part["resolved_side"] = "undetermined"
                    part["resolved_reason"] = "unknown interaction category"
        
        return items

    def _resolve_side_by_min_distance(self, keypoints, object_bboxes, body_part):
        """
        根据身体部位与所有交互物体的最小距离确定是左侧还是右侧。
        
        要求：左右两侧的关键点置信度都必须 >= 0.95，否则标记为无法判定。
        
        Args:
            keypoints: 关键点字典
            object_bboxes: 所有交互物体的 bbox 列表
            body_part: 身体部位名称（如 "wrist", "elbow" 等）
            
        Returns:
            dict: {"side": "left"/"right"/"undetermined", "reason": str, "details": {...}}
        """
        if not object_bboxes:
            return {"side": "undetermined", "reason": "no objects to compare"}
        
        # 获取左右关键点索引
        indices = self.BODY_PART_TO_INDICES.get(body_part.lower())
        if not indices:
            return {"side": "undetermined", "reason": f"unknown body part: {body_part}"}
        
        left_idx, right_idx = indices
        
        # 对于只有一个点的部位（如鼻子），无法判断左右
        if right_idx is None:
            return {"side": "center", "reason": "single point body part (e.g., nose)"}
        
        # 获取左右关键点名称
        left_name = self.KEYPOINT_NAMES[left_idx]
        right_name = self.KEYPOINT_NAMES[right_idx]
        
        left_point = keypoints.get(left_name)
        right_point = keypoints.get(right_name)
        
        # 检查关键点是否存在
        left_exists = left_point is not None and isinstance(left_point, dict)
        right_exists = right_point is not None and isinstance(right_point, dict)
        
        if not left_exists and not right_exists:
            return {"side": "undetermined", "reason": "both keypoints not detected"}
        
        if not left_exists:
            return {"side": "undetermined", "reason": f"left keypoint ({left_name}) not detected"}
        
        if not right_exists:
            return {"side": "undetermined", "reason": f"right keypoint ({right_name}) not detected"}
        
        # 检查置信度是否都 >= 0.95
        left_conf = left_point["confidence"]
        right_conf = right_point["confidence"]
        
        if left_conf < self.CONFIDENCE_THRESHOLD_VALID and right_conf < self.CONFIDENCE_THRESHOLD_VALID:
            return {
                "side": "undetermined", 
                "reason": f"both keypoints low confidence (left: {left_conf:.2f}, right: {right_conf:.2f}, threshold: {self.CONFIDENCE_THRESHOLD_VALID})"
            }
        
        if left_conf < self.CONFIDENCE_THRESHOLD_VALID:
            return {
                "side": "undetermined", 
                "reason": f"left keypoint ({left_name}) low confidence: {left_conf:.2f} < {self.CONFIDENCE_THRESHOLD_VALID}"
            }
        
        if right_conf < self.CONFIDENCE_THRESHOLD_VALID:
            return {
                "side": "undetermined", 
                "reason": f"right keypoint ({right_name}) low confidence: {right_conf:.2f} < {self.CONFIDENCE_THRESHOLD_VALID}"
            }
        
        # 两边置信度都 >= 0.95，计算距离
        left_pos = (left_point["x"], left_point["y"])
        right_pos = (right_point["x"], right_point["y"])
        
        min_dist_left = float('inf')
        min_dist_right = float('inf')
        closest_obj_left = None
        closest_obj_right = None
        
        for obj_idx, obj_bbox in enumerate(object_bboxes):
            # 计算物体中心
            obj_center = (
                (obj_bbox[0] + obj_bbox[2]) / 2,
                (obj_bbox[1] + obj_bbox[3]) / 2
            )
            
            # 计算距离
            dist_left = self._calculate_distance(left_pos, obj_center)
            dist_right = self._calculate_distance(right_pos, obj_center)
            
            if dist_left < min_dist_left:
                min_dist_left = dist_left
                closest_obj_left = obj_idx
            
            if dist_right < min_dist_right:
                min_dist_right = dist_right
                closest_obj_right = obj_idx
        
        # 比较最小距离，选择更近的一侧
        details = {
            "left_keypoint": left_name,
            "right_keypoint": right_name,
            "left_position": left_pos,
            "right_position": right_pos,
            "left_confidence": round(left_conf, 3),
            "right_confidence": round(right_conf, 3),
            "min_distance_left": round(min_dist_left, 2),
            "min_distance_right": round(min_dist_right, 2),
            "closest_object_left": closest_obj_left,
            "closest_object_right": closest_obj_right,
            "num_objects": len(object_bboxes)
        }
        
        if min_dist_left < min_dist_right:
            return {
                "side": "left", 
                "reason": f"left closer to object (dist: {min_dist_left:.2f} < {min_dist_right:.2f})",
                "details": details
            }
        else:
            return {
                "side": "right", 
                "reason": f"right closer to object (dist: {min_dist_right:.2f} < {min_dist_left:.2f})",
                "details": details
            }

    def _calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5
