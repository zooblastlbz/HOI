"""
基于 MPI 的多机多卡 SAM 身体部位 BBox 标注模块（一阶段版本）

使用方法:
    单机多卡:
        mpirun -np 8 python mpi_bodypart_bbox_annotation.py --input_json input.json --output_json output.json
    
    多机多卡:
        mpirun -np 16 --hostfile hostfile python mpi_bodypart_bbox_annotation.py --input_json input.json --output_json output.json

功能说明:
    - 使用 SAM 模型检测图像中人体各个部位的 bounding box
    - 使用 YOLO 的关键点作为 SAM 的辅助提示
    - 一阶段标注，无需人工干预
    - 支持 MPI 多进程并行处理
    - 支持可视化输出（带不透明度的 bbox）
"""

import os
import sys
import time
import json
import math
import shutil
import argparse
from glob import glob
from typing import List, Dict, Optional, Tuple, Any

from regex import F

# ================== MPI 初始化与配置 ==================

def get_mpi_info():
    """通过环境变量获取 MPI rank 和 size"""
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))
    if rank == -1:
        rank = int(os.environ.get('PMI_RANK', -1))
    if rank == -1:
        rank = 0

    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', -1))
    if size == -1:
        size = int(os.environ.get('PMI_SIZE', -1))
    if size == -1:
        size = 1

    return rank, size


RANK, WORLD_SIZE = get_mpi_info()
LOCAL_RANK = RANK % 8 if RANK != -1 else 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)

import cv2
import numpy as np
import torch
from PIL import Image

Sam3Processor = None
Sam3Model = None
SAM_AVAILABLE=False
try:
    from transformers import Sam3Processor, Sam3Model  # type: ignore
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not found or Sam3 not available.")



# ================== SAM 身体部位 BBox 标注类（一阶段版本） ==================

class SamBodyPartAnnotator:
    """SAM 身体部位 BBox 标注器（使用 YOLO 关键点辅助）"""
    
    BODY_PART_NAMES = [
        "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_hand", "right_hand", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_foot", "right_foot"
    ]
    
    BODY_PART_PROMPTS = {
        "left_eye": "left eye", "right_eye": "right eye",
        "left_ear": "left ear", "right_ear": "right ear",
        "left_shoulder": "left shoulder", "right_shoulder": "right shoulder",
        "left_elbow": "left elbow", "right_elbow": "right elbow",
        "left_hand": "left hand", "right_hand": "right hand",
        "left_hip": "left hip", "right_hip": "right hip",
        "left_knee": "left knee", "right_knee": "right knee",
        "left_foot": "left foot", "right_foot": "right foot"
    }
    
    # YOLO 关键点名称映射
    YOLO_KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # 身体部位名称到 YOLO 关键点名称的映射
    BODY_PART_TO_YOLO_KEYPOINT = {
        "left_eye": "left_eye", "right_eye": "right_eye",
        "left_ear": "left_ear", "right_ear": "right_ear",
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow", "right_elbow": "right_elbow",
        "left_hand": "left_wrist", "right_hand": "right_wrist",  # hand 对应 wrist
        "left_hip": "left_hip", "right_hip": "right_hip",
        "left_knee": "left_knee", "right_knee": "right_knee",
        "left_foot": "left_ankle", "right_foot": "right_ankle"  # foot 对应 ankle
    }
    
    SIMPLIFIED_BODY_PARTS = ["left_hand", "right_hand", "left_foot", "right_foot", "head", "torso"]
    
    SIMPLIFIED_PROMPTS = {
        "left_hand": "left hand", "right_hand": "right hand",
        "left_foot": "left foot", "right_foot": "right foot",
        "head": "head", "torso": "torso"
    }

    def __init__(self, model_name: str = "facebook/sam2-hiera-large",
                 conf_threshold: float = 0.5, mask_threshold: float = 0.5, use_simplified_parts: bool = False) -> None:
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.device = "cuda:0"
        
        if use_simplified_parts:
            self.body_parts = self.SIMPLIFIED_BODY_PARTS
            self.prompts = self.SIMPLIFIED_PROMPTS
        else:
            self.body_parts = self.BODY_PART_NAMES
            self.prompts = self.BODY_PART_PROMPTS
        
        self.model = None
        self.processor = None
        
        if SAM_AVAILABLE:
            print(f"[Rank {RANK}] Loading SAM: {model_name}...")
            try:
                self.processor = Sam3Processor.from_pretrained(model_name)
                self.model = Sam3Model.from_pretrained(model_name).to(self.device).eval()
                print(f"[Rank {RANK}] SAM loaded.")
            except Exception as e:
                print(f"[Rank {RANK}] Failed to load SAM: {e}")

    def _convert_bbox(self, bbox: Any) -> Optional[List[int]]:
        if bbox is None:
            return None
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        return [int(b) for b in bbox]

    def _extract_keypoints_from_pose(self, pose_annotation: Optional[Dict]) -> List[Dict[str, Dict[str, Optional[Tuple[float, float]]]]]:
        """
        从 YOLO 姿势标注中提取所有人的关键点坐标
        
        Args:
            pose_annotation: 来自 mpi_pose_annotation.py 的输出
            
        Returns:
            关键点列表，每个元素为一个人的关键点字典
            格式为 [{"person_id": 0, "keypoints": {part_name: (x, y) or None}}, ...]
        """
        persons_keypoints = []
        
        if not pose_annotation or not pose_annotation.get('success'):
            return persons_keypoints
        
        persons = pose_annotation.get('persons', [])
        if not persons:
            return persons_keypoints
        
        # 处理每一个检测到的人
        for person in persons:
            person_id = person.get('person_id', 0)
            person_kpts_data = person.get('keypoints', {})
            
            keypoints = {}
            # 映射 YOLO 关键点到我们的身体部位名称
            for part_name in self.body_parts:
                # 使用映射表获取对应的 YOLO 关键点名称
                yolo_kpt_name = self.BODY_PART_TO_YOLO_KEYPOINT.get(part_name, part_name)
                
                if yolo_kpt_name in person_kpts_data:
                    kpt = person_kpts_data[yolo_kpt_name]
                    if kpt is not None:
                        keypoints[part_name] = (kpt['x'], kpt['y'])
                    else:
                        keypoints[part_name] = None
                else:
                    keypoints[part_name] = None
            
            persons_keypoints.append({
                "person_id": person_id,
                "keypoints": keypoints
            })
        
        return persons_keypoints

    def _extract_person_bboxes(self, pose_annotation: Optional[Dict]) -> Dict[int, List[float]]:
        """
        从 YOLO 姿势标注中提取所有人的 bbox
        
        Args:
            pose_annotation: 来自 mpi_pose_annotation.py 的输出
            
        Returns:
            人物 bbox 字典 {person_id: [x1, y1, x2, y2]}
        """
        bboxes = {}
        
        if not pose_annotation or not pose_annotation.get('success'):
            return bboxes
        
        persons = pose_annotation.get('persons', [])
        for person in persons:
            person_id = person.get('person_id', 0)
            bbox = person.get('bbox')
            if bbox:
                bboxes[person_id] = bbox
        
        return bboxes
    
    def _crop_image_around_keypoint(self, image: Image.Image, keypoint: Tuple[float, float],
                                    crop_ratio: float = 0.2) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        根据关键点裁剪图像的一定比例区域
        
        Args:
            image: PIL 图像
            keypoint: 关键点坐标 (x, y)
            crop_ratio: 裁剪区域大小相对于图像的比例（默认 0.2 = 五分之一）
            
        Returns:
            (裁剪后的图像, 裁剪区域的左上角坐标 (x1, y1))
        """
        x, y = keypoint
        img_width, img_height = image.size
        
        # 计算裁剪区域大小
        crop_width = int(img_width * crop_ratio)
        crop_height = int(img_height * crop_ratio)
        half_width = crop_width / 2
        half_height = crop_height / 2
        
        # 计算裁剪区域的边界
        x1 = max(0, int(x - half_width))
        y1 = max(0, int(y - half_height))
        x2 = min(img_width, int(x + half_width))
        y2 = min(img_height, int(y + half_height))
        
        # 确保裁剪区域大小一致（如果接近边界则调整）
        actual_width = x2 - x1
        actual_height = y2 - y1
        
        if actual_width < crop_width:
            if x1 == 0:
                x2 = min(img_width, x1 + crop_width)
            else:
                x1 = max(0, x2 - crop_width)
        
        if actual_height < crop_height:
            if y1 == 0:
                y2 = min(img_height, y1 + crop_height)
            else:
                y1 = max(0, y2 - crop_height)
        
        # 裁剪图像
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image, (x1, y1)

    def _get_bbox_center(self, bbox: List) -> Tuple[float, float]:
        """计算 bbox 的中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点之间的欧几里得距离"""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def _assign_bbox_to_parts(self, left_part: str, right_part: str,
                             left_result: Optional[Dict], right_result: Optional[Dict],
                             left_keypoint: Optional[Tuple[float, float]],
                             right_keypoint: Optional[Tuple[float, float]]) -> Dict[str, Optional[Dict]]:
        """
        根据 bbox 到关键点的距离判断 bbox 属于左还是右
        同时考虑 YOLO 关键点与 SAM bbox 的一致性
        
        Args:
            left_part: 左部位名称
            right_part: 右部位名称
            left_result: 左部位的检测结果
            right_result: 右部位的检测结果
            left_keypoint: 左部位的关键点
            right_keypoint: 右部位的关键点
            
        Returns:
            分配后的结果字典 {part_name: result}
        """
        results = {}
        
        # 如果两个结果都无效，直接返回
        if not left_result or not right_result:
            results[left_part] = left_result
            results[right_part] = right_result
            return results
        
        # 获取 bbox
        left_bbox = left_result.get('bbox')
        right_bbox = right_result.get('bbox')
        
        if not left_bbox or not right_bbox:
            results[left_part] = left_result
            results[right_part] = right_result
            return results
        
        # 计算 bbox 中心
        left_center = self._get_bbox_center(left_bbox)
        right_center = self._get_bbox_center(right_bbox)
        
        # 情况1：两个关键点都有效
        if left_keypoint and right_keypoint:
            # 计算每个 bbox 到两个关键点的距离
            left_to_left = self._calculate_distance(left_center, left_keypoint)
            left_to_right = self._calculate_distance(left_center, right_keypoint)
            right_to_left = self._calculate_distance(right_center, left_keypoint)
            right_to_right = self._calculate_distance(right_center, right_keypoint)
            
            # 判断每个 bbox 更接近哪个关键点
            left_is_left = left_to_left < left_to_right
            right_is_right = right_to_right < right_to_left
            
            if left_is_left and right_is_right:
                # 判定结果一致，保留两个 bbox
                results[left_part] = left_result
                results[right_part] = right_result
            elif left_is_left and not right_is_right:
                # left_bbox 是左，right_bbox 也是左，只保留更接近左关键点的
                if left_to_left < right_to_left:
                    results[left_part] = left_result
                    results[right_part] = None
                else:
                    results[left_part] = right_result
                    results[right_part] = None
            elif not left_is_left and right_is_right:
                # left_bbox 是右，right_bbox 是右，只保留更接近右关键点的
                if left_to_right < right_to_right:
                    results[left_part] = None
                    results[right_part] = left_result
                else:
                    results[left_part] = None
                    results[right_part] = right_result
            else:
                # 两个都是反的，交换
                results[left_part] = right_result
                results[right_part] = left_result
        
        # 情况2：只有左关键点有效
        elif left_keypoint and not right_keypoint:
            # 如果左 bbox 更接近左关键点，则使用左 bbox；否则舍弃
            left_to_left = self._calculate_distance(left_center, left_keypoint)
            right_to_left = self._calculate_distance(right_center, left_keypoint)
            
            if left_to_left < right_to_left:
                # 左 bbox 更接近左关键点，使用左 bbox
                results[left_part] = left_result
                results[right_part] = None
            else:
                # 右 bbox 更接近左关键点，舍弃两个 bbox
                results[left_part] = None
                results[right_part] = None
        
        # 情况3：只有右关键点有效
        elif right_keypoint and not left_keypoint:
            # 如果右 bbox 更接近右关键点，则使用右 bbox；否则舍弃
            left_to_right = self._calculate_distance(left_center, right_keypoint)
            right_to_right = self._calculate_distance(right_center, right_keypoint)
            
            if right_to_right < left_to_right:
                # 右 bbox 更接近右关键点，使用右 bbox
                results[left_part] = None
                results[right_part] = right_result
            else:
                # 左 bbox 更接近右关键点，舍弃两个 bbox
                results[left_part] = None
                results[right_part] = None
        
        # 情况4：两个关键点都无效
        else:
            results[left_part] = left_result
            results[right_part] = right_result
        
        return results
    
    def _crop_image_by_bbox(self, image: Image.Image, bbox: List[float]) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        根据 bbox 裁剪图像
        
        Args:
            image: PIL 图像
            bbox: bbox 坐标 [x1, y1, x2, y2]
            
        Returns:
            (裁剪后的图像, 裁剪区域的左上角坐标 (x1, y1))
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        
        # 裁剪图像
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image, (x1, y1)
    
    def _detect_part_with_keypoint_bbox(self, image: Image.Image, part: str,
                                       keypoint: Optional[Tuple[float, float]] = None,
                                       person_bbox: Optional[List[float]] = None) -> Optional[Dict]:
        """
        根据 human bbox 裁剪图像，使用文本提示检测身体部位
        
        Args:
            image: PIL 图像
            part: 身体部位名称
            keypoint: 来自 YOLO 的关键点坐标 (x, y)，可选
            person_bbox: 人物 bbox [x1, y1, x2, y2]
            
        Returns:
            检测结果字典或 None（仅当有关键点时才进行检测）
        """
        if self.model is None:
            return None
        
        # 只有当有关键点时才进行检测
        if keypoint is None:
            return None
        
        # 如果没有 person_bbox，返回 None
        if person_bbox is None:
            return None
        
        prompt = self.prompts.get(part, part)
        try:
            # 根据 person_bbox 裁剪图像
            cropped_image, crop_offset = self._crop_image_by_bbox(image, person_bbox)
            crop_x1, crop_y1 = crop_offset
            
            # 计算关键点在裁剪图像中的相对坐标
            keypoint_in_crop = (keypoint[0] - crop_x1, keypoint[1] - crop_y1)
            
            # 使用文本提示作为 SAM 的输入（不使用 bbox 输入）
            inputs = self.processor(
                images=cropped_image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=[(cropped_image.height, cropped_image.width)],
                threshold=self.conf_threshold, mask_threshold=self.mask_threshold
            )
            
            if results and len(results[0].get('boxes', [])) > 0:
                print(f"[Rank {RANK}] Detected {len(results[0]['boxes'])} boxes for part: {part}")
                boxes, scores = results[0]['boxes'], results[0].get('scores', [])
                
                # 如果有多个检测结果，选择与关键点距离最近的
                if len(boxes) > 1:
                    min_distance = float('inf')
                    best_idx = 0
                    for idx, bbox in enumerate(boxes):
                        bbox_converted = self._convert_bbox(bbox)
                        if bbox_converted:
                            bbox_center = self._get_bbox_center(bbox_converted)
                            distance = self._calculate_distance(keypoint_in_crop, bbox_center)
                            if distance < min_distance:
                                min_distance = distance
                                best_idx = idx
                    print(f"[Rank {RANK}] Selected bbox {best_idx} (distance: {min_distance:.2f}) for part: {part}")
                else:
                    best_idx = 0
                
                detected_bbox = self._convert_bbox(boxes[best_idx])
                if detected_bbox:
                    # 将裁剪图像中的坐标转换回原始图像坐标
                    original_bbox = [
                        detected_bbox[0] + crop_x1,
                        detected_bbox[1] + crop_y1,
                        detected_bbox[2] + crop_x1,
                        detected_bbox[3] + crop_y1
                    ]
                    return {"bbox": original_bbox,
                            "confidence": float(scores[best_idx]) if scores else 1.0}
            print(f"[Rank {RANK}] No boxes detected for part: {part}")
            return {"error": "No boxes detected by SAM"}
        except Exception as e:
            print(f"[Rank {RANK}] Error detecting {part}: {e}")
            return {"error": f"SAM detection error: {str(e)}"}
    
    def _detect_part_with_keypoint(self, image: Image.Image, part: str,
                                   keypoint: Optional[Tuple[float, float]] = None,
                                   crop_ratio: float = 0.2) -> Optional[Dict]:
        """
        根据关键点裁剪图像，使用文本提示检测身体部位
        
        Args:
            image: PIL 图像
            part: 身体部位名称
            keypoint: 来自 YOLO 的关键点坐标 (x, y)，可选
            crop_ratio: 裁剪区域大小相对于图像的比例（默认 0.2 = 五分之一）
            
        Returns:
            检测结果字典或 None（仅当有关键点时才进行检测）
        """
        if self.model is None:
            return None
        
        # 只有当有关键点时才进行检测
        if keypoint is None:
            return None
        
        prompt = self.prompts.get(part, part)
        try:
            # 根据关键点裁剪图像
            cropped_image, crop_offset = self._crop_image_around_keypoint(image, keypoint, crop_ratio)
            crop_x1, crop_y1 = crop_offset
            
            # 计算关键点在裁剪图像中的相对坐标
            keypoint_in_crop = (keypoint[0] - crop_x1, keypoint[1] - crop_y1)
            
            # 使用文本提示作为 SAM 的输入（不使用 bbox 输入）
            inputs = self.processor(
                images=cropped_image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=[(cropped_image.height, cropped_image.width)],
                threshold=self.conf_threshold, mask_threshold=self.mask_threshold
            )
            
            if results and len(results[0].get('boxes', [])) > 0:
                print(f"[Rank {RANK}] Detected {len(results[0]['boxes'])} boxes for part: {part}")
                boxes, scores = results[0]['boxes'], results[0].get('scores', [])
                
                # 选择置信度最高的 bbox
                best_idx = 0
                if len(boxes) > 1 and scores:
                    best_idx = int(np.argmax(scores))
                    print(f"[Rank {RANK}] Selected bbox {best_idx} (confidence: {scores[best_idx]:.4f}) for part: {part}")
                
                detected_bbox = self._convert_bbox(boxes[best_idx])
                if detected_bbox:
                    # 将裁剪图像中的坐标转换回原始图像坐标
                    original_bbox = [
                        detected_bbox[0] + crop_x1,
                        detected_bbox[1] + crop_y1,
                        detected_bbox[2] + crop_x1,
                        detected_bbox[3] + crop_y1
                    ]
                    return {"bbox": original_bbox,
                            "confidence": float(scores[best_idx]) if scores else 1.0}
            print(f"[Rank {RANK}] No boxes detected for part: {part}")
            return {"error": "No boxes detected by SAM"}
        except Exception as e:
            print(f"[Rank {RANK}] Error detecting {part}: {e}")
            return {"error": f"SAM detection error: {str(e)}"}

    def process_single_image(self, image_path: str, pose_annotation: Optional[Dict] = None,
                            visualize: bool = False, vis_output_dir: Optional[str] = None,
                            vis_opacity: float = 0.3, vis_line_width: int = 2, sample_idx: int = 0,
                            crop_ratio: float = 0.2) -> Dict:
        """
        处理单张图片，根据 human bbox 裁剪图像后进行标注
        
        Args:
            image_path: 图片路径
            pose_annotation: YOLO 姿势标注结果（包含 human bbox）
            visualize: 是否生成可视化
            vis_output_dir: 可视化输出目录
            vis_opacity: 可视化透明度
            vis_line_width: 可视化线宽
            sample_idx: 样本索引
            crop_ratio: 保留用于兼容性，但不再使用
            
        Returns:
            标注结果字典
        """
        if not self.model:
            return {"success": False, "error": "SAM not loaded"}
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}
        
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            
            # 从 YOLO 标注中提取所有人的关键点和 bbox
            persons_keypoints = self._extract_keypoints_from_pose(pose_annotation)
            persons_bboxes = self._extract_person_bboxes(pose_annotation)
            
            # 处理每一个人的关键点
            persons_results = []
            for person_data in persons_keypoints:
                person_id = person_data["person_id"]
                person_keypoints = person_data["keypoints"]
                person_bbox = persons_bboxes.get(person_id)
                
                # 使用关键点辅助检测各身体部位
                body_parts_results = {}
                
                # 分组处理成对的部位（left/right）
                processed_parts = set()
                for part in self.body_parts:
                    if part in processed_parts:
                        continue
                    
                    # 检查是否是成对的部位
                    if part.startswith('left_'):
                        base_part = part[5:]  # 移除 'left_' 前缀
                        right_part = f'right_{base_part}'
                        
                        if right_part in self.body_parts:
                            # 这是一个成对的部位，同时检测 left 和 right
                            left_keypoint = person_keypoints.get(part)
                            right_keypoint = person_keypoints.get(right_part)
                            
                            # 检测两个部位（使用 person_bbox 而不是 crop_ratio）
                            left_result = self._detect_part_with_keypoint_bbox(image, part, left_keypoint, person_bbox)
                            right_result = self._detect_part_with_keypoint_bbox(image, right_part, right_keypoint, person_bbox)
                            
                            # 根据 bbox 到关键点的距离判断左右
                            assigned_results = self._assign_bbox_to_parts(
                                part, right_part, left_result, right_result,
                                left_keypoint, right_keypoint
                            )
                            body_parts_results.update(assigned_results)
                            
                            processed_parts.add(part)
                            processed_parts.add(right_part)
                        else:
                            # 不是成对的部位，单独处理
                            keypoint = person_keypoints.get(part)
                            result = self._detect_part_with_keypoint_bbox(image, part, keypoint, person_bbox)
                            body_parts_results[part] = result
                            processed_parts.add(part)
                    elif not part.startswith('right_'):
                        # 非左右配对的部位（如 head, torso 等）
                        keypoint = person_keypoints.get(part)
                        result = self._detect_part_with_keypoint_bbox(image, part, keypoint, person_bbox)
                        body_parts_results[part] = result
                        processed_parts.add(part)
                
                # 仅保存成功的 bbox，失败的部分不保存
                successful_bboxes = {}
                for part_name, part_result in body_parts_results.items():
                    if part_result is not None and 'bbox' in part_result:
                        successful_bboxes[part_name] = part_result['bbox']
                
                valid_count = len(successful_bboxes)
                
                persons_results.append({
                    "person_id": person_id,
                    "bbox": successful_bboxes,
                    "valid_parts_count": valid_count,
                    "total_parts": len(self.body_parts)
                })
            
            result = {"success": True, "image_size": {"width": w, "height": h},
                     "persons": persons_results, "num_persons": len(persons_results)}
            
            # 如果启用可视化，立即生成可视化图像
            if visualize and vis_output_dir:
                vis_dir = os.path.join(vis_output_dir, f"rank_{RANK}")
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"vis_{sample_idx}_{os.path.basename(image_path)}")
                success = visualize_body_part_bboxes(image_path, result, vis_path, vis_opacity, vis_line_width)
                if success:
                    print(f"[Rank {RANK}] Saved visualization: {vis_path}")
                    result["visualization_path"] = vis_path
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_batch(self, image_paths: List[str], pose_annotations: Optional[List[Optional[Dict]]] = None,
                     visualize: bool = False, vis_output_dir: Optional[str] = None,
                     vis_opacity: float = 0.3, vis_line_width: int = 2, start_sample_idx: int = 0,
                     crop_ratio: float = 0.2) -> List[Dict]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            pose_annotations: YOLO 姿势标注列表
            visualize: 是否生成可视化
            vis_output_dir: 可视化输出目录
            vis_opacity: 可视化透明度
            vis_line_width: 可视化线宽
            start_sample_idx: 起始样本索引
            crop_ratio: 裁剪区域大小相对于图像的比例（默认 0.2 = 十分之一）
            
        Returns:
            结果列表
        """
        if pose_annotations is None:
            pose_annotations = [None for _ in image_paths]
        
        results = []
        for i, (img_path, pose_anno) in enumerate(zip(image_paths, pose_annotations)):
            result = self.process_single_image(
                img_path, pose_anno, visualize, vis_output_dir, vis_opacity, vis_line_width,
                sample_idx=start_sample_idx + i, crop_ratio=crop_ratio
            )
            results.append(result)
        return results

    def annotate_json_data(self, data_list: List[Dict], image_key: str = "image_path",
                          pose_key: str = "pose_annotation",
                          output_key: str = "bodypart_bbox_annotation", visualize: bool = False,
                          vis_output_dir: Optional[str] = None, vis_opacity: float = 0.3, vis_line_width: int = 2,
                          batch_size: int = 10, temp_dir: Optional[str] = None, crop_ratio: float = 0.2) -> List[Dict]:
        """
        为 JSON 数据列表添加身体部位 BBox 标注
        
        Args:
            data_list: 数据列表
            image_key: 图片路径的 key
            pose_key: YOLO 姿势标注的 key
            output_key: 输出 BBox 标注的 key
            visualize: 是否生成可视化
            vis_output_dir: 可视化输出目录
            vis_opacity: 可视化透明度
            vis_line_width: 可视化线宽
            batch_size: 批处理大小
            temp_dir: 临时目录
            crop_ratio: 裁剪区域大小相对于图像的比例（默认 0.2 = 十分之一）
            
        Returns:
            标注后的数据列表
        """
        image_paths, path_to_indices = [], {}
        for i, item in enumerate(data_list):
            img_path = item.get(image_key)
            if img_path:
                if img_path not in path_to_indices:
                    path_to_indices[img_path] = []
                    image_paths.append(img_path)
                path_to_indices[img_path].append(i)
        
        print(f"[Rank {RANK}] Processing {len(image_paths)} images in batches of {batch_size}...")
        
        # 分批处理图片
        for batch_idx, start_idx in enumerate(range(0, len(image_paths), batch_size)):
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_image_paths = image_paths[start_idx:end_idx]
            
            # 收集对应的 YOLO 姿势标注
            batch_pose_annotations = []
            for img_path in batch_image_paths:
                # 获取该图片对应的第一个数据项的 YOLO 标注
                first_idx = path_to_indices[img_path][0]
                pose_anno = data_list[first_idx].get(pose_key)
                batch_pose_annotations.append(pose_anno)
            
            print(f"[Rank {RANK}] Processing batch {batch_idx + 1}: images {start_idx}-{end_idx-1}")
            
            batch_results = self.process_batch(
                batch_image_paths, batch_pose_annotations, visualize, vis_output_dir,
                vis_opacity, vis_line_width, start_sample_idx=start_idx, crop_ratio=crop_ratio
            )
            
            # 将结果应用到数据列表
            for img_path, result in zip(batch_image_paths, batch_results):
                for idx in path_to_indices[img_path]:
                    data_list[idx][output_key] = result
                    # 若开启可视化，将 image_path 替换为可视化文件路径
                    vis_path = result.get("visualization_path")
                    if visualize and vis_path:
                        data_list[idx][image_key] = vis_path
            
            # 每处理完一个batch就保存临时结果
            if temp_dir:
                batch_temp_file = os.path.join(temp_dir, f"rank_{RANK}_batch_{batch_idx}.json")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 保存当前batch的结果
                batch_data = []
                for img_path in batch_image_paths:
                    for idx in path_to_indices[img_path]:
                        batch_data.append(data_list[idx])
                
                with open(batch_temp_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                print(f"[Rank {RANK}] Saved batch {batch_idx + 1} to {batch_temp_file}")
        
        for item in data_list:
            if output_key not in item:
                item[output_key] = {"success": False, "error": f"No {image_key}"}
        return data_list


# ================== 可视化函数 ==================

def draw_bbox_with_opacity(img: np.ndarray, bbox: List[int], color: Tuple[int, int, int],
                           opacity: float = 0.7, line_width: int = 3) -> np.ndarray:
    """
    绘制带透明度的 bbox 边框
    
    Args:
        img: 输入图像
        bbox: bbox 坐标 [x1, y1, x2, y2]
        color: 边框颜色 (B, G, R)
        opacity: 边框透明度（默认 0.7 = 70% 不透明，更显眼）
        line_width: 边框线宽（默认 3 像素）
        
    Returns:
        绘制后的图像
    """
    x1, y1, x2, y2 = bbox
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_width)
    img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    return img


def visualize_body_part_bboxes(image_path: str, annotation: Dict, output_path: str,
                                  opacity: float = 0.3, line_width: int = 2) -> bool:
      """
      可视化身体部位的 bbox 标注
      
      注意：标记的是 SAM 检测输出的 bbox，而不是输入给 SAM 的 bbox
      - 输入 bbox：由 YOLO 关键点生成的小 bbox（用于引导 SAM）
      - 输出 bbox：SAM 检测到的身体部位的实际 bbox（这是标记的内容）
      
      颜色编码：
      - 绿色 (0, 255, 0)：左侧身体部位 (left_*)
      - 蓝色 (0, 0, 255)：右侧身体部位 (right_*)
      - 红色 (255, 0, 0)：中心身体部位 (其他)
      """
      try:
          img = cv2.imread(image_path)
          if img is None:
              return False
          
          # 使用绿色和红色标记左右，蓝色标记中心
          colors = {'left': (0, 255, 0), 'right': (0, 0, 255), 'center': (255, 0, 0)}
          
          # 处理多个人的情况
          persons = annotation.get('persons', [])
          if persons:
              # 新格式：多个人
              for person in persons:
                  bboxes = person.get('bbox', {})
                  for part_name, bbox in bboxes.items():
                      if bbox is None:
                          continue
                      
                      if part_name.startswith('left'):
                          color = colors['left']
                      elif part_name.startswith('right'):
                          color = colors['right']
                      else:
                          color = colors['center']
                      
                      img = draw_bbox_with_opacity(img, bbox, color, opacity, line_width)
          else:
              # 旧格式：单个人（向后兼容）
              body_parts = annotation.get('body_parts', {})
              for part_name, part_info in body_parts.items():
                  if part_info is None or part_info.get('bbox') is None:
                      continue
                  
                  if part_name.startswith('left'):
                      color = colors['left']
                  elif part_name.startswith('right'):
                      color = colors['right']
                  else:
                      color = colors['center']
                  
                  img = draw_bbox_with_opacity(img, part_info['bbox'], color, opacity, line_width)
          
          os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
          cv2.imwrite(output_path, img)
          return True
      except:
          return False


# ================== 主程序 ==================

def update_image_paths(input_json: str, output_json: str, image_key: str = "image_path",
                       vis_output_dir: str = None) -> None:
    """
    更新输出 JSON 文件中的图像路径，使用可视化输出目录中的图像
    支持 MPI 环境下的多进程可视化文件收集
    
    Args:
        input_json: 输入 JSON 文件路径
        output_json: 输出 JSON 文件路径
        image_key: 图像路径的 key
        vis_output_dir: 可视化输出目录
    """
    if not vis_output_dir:
        print("Error: vis_output_dir is required for updating image paths")
        return
    
    # 读取输入文件获取原始图像路径
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 读取输出文件
    with open(output_json, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    # 收集所有可视化文件（支持 MPI 多进程）
    # 在 MPI 环境下，可视化文件分散在各个 rank 目录中
    all_vis_files = {}
    for rank_dir in glob(os.path.join(vis_output_dir, "rank_*")):
        for vis_file in glob(os.path.join(rank_dir, "vis_*")):
            # 提取文件名中的索引和原始文件名
            # 文件格式：vis_{idx}_{original_filename}
            basename = os.path.basename(vis_file)
            if basename.startswith("vis_"):
                # 移除 "vis_" 前缀
                parts = basename[4:].split("_", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        original_filename = parts[1]
                        if idx not in all_vis_files:
                            all_vis_files[idx] = {}
                        all_vis_files[idx][original_filename] = vis_file
                    except ValueError:
                        pass
    
    # 创建原始路径到可视化路径的映射
    path_mapping = {}
    for idx, item in enumerate(input_data):
        original_path = item.get(image_key)
        if original_path:
            filename = os.path.basename(original_path)
            # 查找对应的可视化文件
            if idx in all_vis_files and filename in all_vis_files[idx]:
                path_mapping[original_path] = all_vis_files[idx][filename]
    
    # 更新输出文件中的图像路径
    updated_count = 0
    for item in output_data:
        original_path = item.get(image_key)
        if original_path in path_mapping:
            item[image_key] = path_mapping[original_path]
            updated_count += 1
    
    # 保存更新后的输出文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated {updated_count} image paths in {output_json}")
    print(f"Total collected visualization files: {sum(len(v) for v in all_vis_files.values())}")


def main():
    parser = argparse.ArgumentParser(description="MPI SAM Body Part BBox Annotation (One-Stage with YOLO Keypoints)")
    parser.add_argument("--model_name", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/Models/sam3/")
    parser.add_argument("--input_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter_anno.json")
    parser.add_argument("--output_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter_anno_sam.json")
    parser.add_argument("--temp_dir", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter_anno_sam_tmp")
    parser.add_argument("--image_key", type=str, default="image_path")
    parser.add_argument("--pose_key", type=str, default="pose_annotation", help="JSON key for YOLO pose annotation")
    parser.add_argument("--output_key", type=str, default="bodypart_bbox_annotation")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="SAM confidence threshold (default 0.4, lower for more detections)")
    parser.add_argument("--mask_threshold", type=float, default=0.6, help="SAM mask threshold for post-processing (default 0.5)")
    parser.add_argument("--crop_ratio", type=float, default=1.0, help="Crop ratio relative to image size (default 0.3 = 1/3, larger crop for better context)")
    parser.add_argument("--use_simplified_parts", action="store_true")
    parser.add_argument("--visualize",default=True ,action="store_true")
    parser.add_argument("--vis_output_dir", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter_anno_sam")
    parser.add_argument("--vis_opacity", type=float, default=0.6, help="Visualization opacity (default 0.7 = 70% opaque, more visible)")
    parser.add_argument("--vis_line_width", type=int, default=3, help="Visualization line width in pixels (default 4)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous interrupted run")
    parser.add_argument("--force_restart", action="store_true", help="Force restart even if temp files exist")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--update_paths_only", action="store_true", help="Only update image paths in output JSON without re-annotating")
    args = parser.parse_args()
    
    # 如果只需要更新路径，直接调用 update_image_paths 并退出
    if args.update_paths_only:
        if RANK == 0:
            print(f"{'='*60}\nUpdating image paths only\n{'='*60}")
            update_image_paths(args.input_json, args.output_json, args.image_key, args.vis_output_dir)
        return
    
    if RANK == 0:
        print(f"{'='*60}\nMPI SAM Body Part BBox Annotation (One-Stage)\nWorld Size: {WORLD_SIZE}")
        print(f"Input: {args.input_json}\nOutput: {args.output_json}\n{'='*60}")
        if not os.path.exists(args.input_json):
            print(f"Error: {args.input_json} not found.")
            sys.exit(1)
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total = len(all_data)
    chunk_size = math.ceil(total / WORLD_SIZE)
    start_idx, end_idx = RANK * chunk_size, min((RANK + 1) * chunk_size, total)
    my_data = all_data[start_idx:end_idx]
    
    # 检查是否需要resume
    temp_marker_dir = os.path.join(args.temp_dir, "markers")
    temp_json_pattern = os.path.join(args.temp_dir, "rank_*.json")
    temp_marker_pattern = os.path.join(temp_marker_dir, "rank_*.done")
    
    existing_temp_files = glob(temp_json_pattern)
    existing_markers = glob(temp_marker_pattern)
    
    if args.force_restart:
        print(f"[Rank {RANK}] Force restart: cleaning temp directory...")
        shutil.rmtree(args.temp_dir, ignore_errors=True)
        existing_temp_files = []
        existing_markers = []
    elif args.resume and existing_temp_files:
        print(f"[Rank {RANK}] Resume mode: found {len(existing_temp_files)} existing temp files")
        if RANK == 0:
            print(f"[Rank 0] Existing completed ranks: {[f.split('_')[-1].split('.')[0] for f in existing_markers]}")
    
    # 检查当前rank是否已经完成
    my_temp_file = os.path.join(args.temp_dir, f"rank_{RANK}.json")
    my_marker_file = os.path.join(temp_marker_dir, f"rank_{RANK}.done")
    rank_batch_pattern = os.path.join(args.temp_dir, f"rank_{RANK}_batch_*.json")
    existing_rank_batches = sorted(glob(rank_batch_pattern))
    processed_batches = set()
    processed_batch_data = []
    
    for bf in existing_rank_batches:
        try:
            with open(bf, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                processed_batch_data.extend(batch_data)
        except Exception as e:
            print(f"[Rank {RANK}] Warning: failed to load batch file {bf}: {e}")
        # 从文件名中解析 batch_idx
        try:
            fname = os.path.basename(bf)
            batch_idx = int(fname.split("_batch_")[1].split(".")[0])
            processed_batches.add(batch_idx)
        except Exception:
            pass
    
    if args.resume and os.path.exists(my_temp_file) and os.path.exists(my_marker_file):
        print(f"[Rank {RANK}] Already completed, skipping...")
        my_data = []  # 不需要处理
    else:
        # 正常或部分resume处理流程
        print(f"[Rank {RANK}] Processing {len(my_data)} samples ({start_idx}-{end_idx})")
        if args.resume and processed_batches:
            print(f"[Rank {RANK}] Resume: found {len(processed_batches)} completed batches, will skip them.")
        
        rank_results = list(processed_batch_data)  # 先汇总已有的batch结果
        
        if my_data:
            annotator = SamBodyPartAnnotator(args.model_name, args.conf_threshold, args.mask_threshold,
                                            args.use_simplified_parts)
            
            # 基于 annotate_json_data 的逻辑，按 batch 跳过已完成的 batch
            image_paths, path_to_indices = [], {}
            for i, item in enumerate(my_data):
                img_path = item.get(args.image_key)
                if img_path:
                    if img_path not in path_to_indices:
                        path_to_indices[img_path] = []
                        image_paths.append(img_path)
                    path_to_indices[img_path].append(i)
            
            total_batches = math.ceil(len(image_paths) / args.batch_size) if args.batch_size > 0 else 0
            
            start_time = time.time()
            for batch_idx, start_i in enumerate(range(0, len(image_paths), args.batch_size)):
                end_i = min(start_i + args.batch_size, len(image_paths))
                if args.resume and batch_idx in processed_batches:
                    # 已完成的 batch 跳过
                    continue
                
                batch_image_paths = image_paths[start_i:end_i]
                batch_pose_annotations = []
                for img_path in batch_image_paths:
                    first_idx = path_to_indices[img_path][0]
                    pose_anno = my_data[first_idx].get(args.pose_key)
                    batch_pose_annotations.append(pose_anno)
                
                batch_results = annotator.process_batch(
                    batch_image_paths, batch_pose_annotations, args.visualize, args.vis_output_dir,
                    args.vis_opacity, args.vis_line_width, start_sample_idx=start_i, crop_ratio=args.crop_ratio
                )
                
                batch_data = []
                for img_path, result in zip(batch_image_paths, batch_results):
                    for idx in path_to_indices[img_path]:
                        item = dict(my_data[idx])
                        item[args.output_key] = result
                        # 若开启可视化，将 image_path 指向可视化文件
                        vis_path = result.get("visualization_path")
                        if args.visualize and vis_path:
                            item[args.image_key] = vis_path
                        batch_data.append(item)
                
                # 保存该 batch
                if args.temp_dir:
                    batch_temp_file = os.path.join(args.temp_dir, f"rank_{RANK}_batch_{batch_idx}.json")
                    os.makedirs(args.temp_dir, exist_ok=True)
                    with open(batch_temp_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                    print(f"[Rank {RANK}] Saved batch {batch_idx + 1}/{total_batches} to {batch_temp_file}")
                
                rank_results.extend(batch_data)
            
            print(f"[Rank {RANK}] Annotation done in {time.time()-start_time:.2f}s")
        
        # 保存临时结果（已完成的batch + 新处理的batch）
        os.makedirs(args.temp_dir, exist_ok=True)
        with open(my_temp_file, 'w', encoding='utf-8') as f:
            json.dump(rank_results, f, ensure_ascii=False, indent=2)
        
        os.makedirs(temp_marker_dir, exist_ok=True)
        with open(my_marker_file, 'w') as f:
            f.write(f"done at {time.time()}")
        
        print(f"[Rank {RANK}] Saved results to {my_temp_file}")
    
    if RANK == 0:
        print("[Rank 0] Waiting for other processes...")
        while len(glob(os.path.join(temp_marker_dir, "*.done"))) < WORLD_SIZE:
            time.sleep(5)
        
        print("[Rank 0] Merging results...")
        final_data = []
        for r in range(WORLD_SIZE):
            # 收集该rank的所有batch文件
            batch_pattern = os.path.join(args.temp_dir, f"rank_{r}_batch_*.json")
            batch_files = sorted(glob(batch_pattern))
            
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        final_data.extend(batch_data)
            
            # 如果没有batch文件，尝试读取旧格式的文件（向后兼容）
            if not batch_files:
                t_file = os.path.join(args.temp_dir, f"rank_{r}.json")
                if os.path.exists(t_file):
                    with open(t_file, 'r') as f:
                        final_data.extend(json.load(f))
        
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        success = sum(1 for item in final_data if item.get(args.output_key, {}).get('success'))
        print(f"[Rank 0] Output: {args.output_json}")
        print(f"[Rank 0] Success: {success}/{len(final_data)} ({100*success/len(final_data):.2f}%)")
        
        shutil.rmtree(args.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
