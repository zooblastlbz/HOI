"""
基于 MPI 的多机多卡 YOLO 人体姿势标注模块

使用方法:
    单机多卡:
        mpirun -np 8 python mpi_pose_annotation.py --input_json input.json --output_json output.json
    
    多机多卡 (假设每台机器 8 张卡):
        mpirun -np 16 --hostfile hostfile python mpi_pose_annotation.py --input_json input.json --output_json output.json
"""

import os
import sys
import time
import json
import math
import shutil
import argparse
from glob import glob

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
LOCAL_RANK = RANK % 8 if RANK != -1 else 0  # 假设每台机器最多 8 张 GPU

# 关键：强制设置当前进程可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print(f"[Rank {RANK}] Warning: 'ultralytics' package not installed.")


# ================== YOLO Pose 标注类 ==================

class YoloPoseAnnotator:
    """YOLO 人体姿势标注器"""
    
    # YOLO-Pose Keypoint Mapping (COCO format)
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
    
    # 骨骼连接关系，用于可视化
    SKELETON = [
        (0, 1), (0, 2),     # 鼻子到眼睛
        (1, 3), (2, 4),     # 眼睛到耳朵
        (5, 6),             # 左肩到右肩
        (5, 7), (7, 9),     # 左臂
        (6, 8), (8, 10),    # 右臂
        (5, 11), (6, 12),   # 肩到髋
        (11, 12),           # 左髋到右髋
        (11, 13), (13, 15), # 左腿
        (12, 14), (14, 16)  # 右腿
    ]
    
    # 不要在模块顶层设置 CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)  # 删除这行

    # 在 YoloPoseAnnotator.__init__ 中直接指定设备
    def __init__(self, model_path="yolo11n-pose.pt", batch_size=16, conf_threshold=0.5):
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        
        # 直接使用 LOCAL_RANK 指定 GPU 设备
        self.device = f"cuda:0"
        
        if YOLO:
            print(f"[Rank {RANK}] Loading YOLO-Pose model on {self.device}...")
            self.model = YOLO(model_path)
            self.model.to(self.device)  # 显式移动模型到指定 GPU
        else:
            print(f"[Rank {RANK}] Warning: YOLO not available, pose estimation will fail.")
            self.model = None
    
    def _extract_all_persons_keypoints(self, result):
        """
        从 YOLO 结果中提取所有检测到的人的关键点
        
        Args:
            result: YOLO 推理结果
            
        Returns:
            List of person dicts, each containing keypoints and bbox
        """
        persons = []
        
        if result is None or result.keypoints is None:
            return persons
        
        # 获取关键点数据: (N, 17, 3) - N个人, 17个关键点, (x, y, conf)
        kpts_data = result.keypoints.data.cpu().numpy()
        
        # 获取检测框数据
        boxes_data = result.boxes.data.cpu().numpy() if result.boxes is not None else None
        
        if len(kpts_data) == 0:
            return persons
        
        for person_idx in range(len(kpts_data)):
            person_kpts = kpts_data[person_idx]  # shape: (17, 3)
            
            # 提取关键点
            keypoints = {}
            valid_keypoints_count = 0
            
            for i, name in enumerate(self.KEYPOINT_NAMES):
                x, y, conf = person_kpts[i]
                
                if conf >= self.conf_threshold and not (x == 0 and y == 0):
                    keypoints[name] = {
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf)
                    }
                    valid_keypoints_count += 1
                else:
                    keypoints[name] = None
            
            # 获取检测框
            bbox = None
            box_conf = 0.0
            if boxes_data is not None and person_idx < len(boxes_data):
                box = boxes_data[person_idx]
                bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                box_conf = float(box[4]) if len(box) > 4 else 0.0
            
            person_data = {
                "person_id": person_idx,
                "bbox": bbox,
                "bbox_confidence": box_conf,
                "keypoints": keypoints,
                "valid_keypoints_count": valid_keypoints_count,
                "total_keypoints": len(self.KEYPOINT_NAMES)
            }
            
            persons.append(person_data)
        
        return persons
    
    def process_single_image(self, image_path):
        """
        处理单张图片，返回所有检测到的人的姿势信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含处理结果的字典
        """
        if not self.model:
            return {"success": False, "error": "YOLO model not loaded"}
        
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}
        
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": f"Failed to read image: {image_path}"}
            
            h, w = img.shape[:2]
            
            # 运行推理
            results = self.model(img, device=self.device, verbose=False)
            
            if not results:
                return {
                    "success": True,
                    "image_size": {"width": w, "height": h},
                    "persons": [],
                    "num_persons": 0
                }
            
            # 提取所有人的关键点
            persons = self._extract_all_persons_keypoints(results[0])
            
            return {
                "success": True,
                "image_size": {"width": w, "height": h},
                "persons": persons,
                "num_persons": len(persons)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_batch(self, image_paths):
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            List of results
        """
        if not self.model:
            return [{"success": False, "error": "YOLO model not loaded"} for _ in image_paths]
        
        results_list = []
        
        # 分批处理
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            batch_sizes = []
            valid_indices = []
            batch_errors = {}
            
            # 读取图像
            for j, img_path in enumerate(batch_paths):
                if not os.path.exists(img_path):
                    batch_errors[j] = f"Image not found: {img_path}"
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    batch_errors[j] = f"Failed to read image: {img_path}"
                    continue
                
                h, w = img.shape[:2]
                batch_images.append(img)
                batch_sizes.append({"width": w, "height": h})
                valid_indices.append(j)
            
            # 批量推理
            batch_results = [{} for _ in batch_paths]
            
            if batch_images:
                try:
                    yolo_results = self.model(batch_images, device=self.device, verbose=False)
                    
                    for k, valid_idx in enumerate(valid_indices):
                        if k < len(yolo_results):
                            persons = self._extract_all_persons_keypoints(yolo_results[k])
                            batch_results[valid_idx] = {
                                "success": True,
                                "image_size": batch_sizes[k],
                                "persons": persons,
                                "num_persons": len(persons)
                            }
                        else:
                            batch_results[valid_idx] = {
                                "success": True,
                                "image_size": batch_sizes[k],
                                "persons": [],
                                "num_persons": 0
                            }
                            
                except Exception as e:
                    for valid_idx in valid_indices:
                        batch_results[valid_idx] = {"success": False, "error": str(e)}
            
            # 填充错误结果
            for j, error in batch_errors.items():
                batch_results[j] = {"success": False, "error": error}
            
            results_list.extend(batch_results)
        
        return results_list
    
    def annotate_json_data(self, data_list, image_key="image_path", output_key="pose_annotation"):
        """
        为 JSON 数据列表添加姿势标注
        
        Args:
            data_list: JSON 数据列表
            image_key: 图片路径对应的 key
            output_key: 输出姿势标注的 key
            
        Returns:
            标注后的数据列表
        """
        # 收集所有图片路径
        image_paths = []
        path_to_indices = {}  # 映射图片路径到数据索引（处理重复）
        
        for i, item in enumerate(data_list):
            img_path = item.get(image_key)
            if img_path:
                if img_path not in path_to_indices:
                    path_to_indices[img_path] = []
                    image_paths.append(img_path)
                path_to_indices[img_path].append(i)
        
        print(f"[Rank {RANK}] Processing {len(image_paths)} unique images for {len(data_list)} entries...")
        
        # 批量处理
        pose_results = self.process_batch(image_paths)
        
        # 将结果写回数据
        for img_path, result in zip(image_paths, pose_results):
            for idx in path_to_indices[img_path]:
                data_list[idx][output_key] = result
        
        # 处理没有图片路径的数据
        for i, item in enumerate(data_list):
            if output_key not in item:
                item[output_key] = {
                    "success": False,
                    "error": f"No {image_key} found"
                }
        
        return data_list


# ================== 主程序 ==================

def main():
    parser = argparse.ArgumentParser(description="MPI-based Multi-GPU YOLO Pose Annotation")
    parser.add_argument("--model_path", type=str, default="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/yolo11x-pose.pt",
                        help="YOLO Pose model path")
    parser.add_argument("--input_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter.json",
                        help="Input JSON file path")
    parser.add_argument("--output_json", type=str, default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter_anno.json",
                        help="Final Output JSON file path")
    parser.add_argument("--temp_dir", type=str, default="./temp_pose_results",
                        help="Temp directory for intermediate results")
    parser.add_argument("--image_key", type=str, default="image_path",
                        help="JSON key for image path")
    parser.add_argument("--output_key", type=str, default="pose_annotation",
                        help="JSON key for output pose annotation")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for YOLO inference")
    parser.add_argument("--conf_threshold", type=float, default=0.6,
                        help="Confidence threshold for keypoints")
    
    args = parser.parse_args()
    
    if RANK == 0:
        print(f"=" * 60)
        print(f"MPI YOLO Pose Annotation")
        print(f"World Size: {WORLD_SIZE}")
        print(f"Input: {args.input_json}")
        print(f"Output: {args.output_json}")
        print(f"=" * 60)
        
        if not os.path.exists(args.input_json):
            print(f"Error: Input file {args.input_json} not found.")
            sys.exit(1)
    
    # 1. 读取数据
    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_entries = len(all_data)
    
    if RANK == 0:
        print(f"Total entries: {total_entries}")
    
    # 2. 数据分片
    samples_per_process = math.ceil(total_entries / WORLD_SIZE)
    start_idx = RANK * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_entries)
    
    my_data = all_data[start_idx:end_idx]
    
    print(f"[Rank {RANK}] Processing {len(my_data)} samples (index {start_idx} to {end_idx})")
    
    # 3. 如果当前 Rank 有数据，初始化模型并处理
    if len(my_data) > 0:
        # 初始化标注器
        annotator = YoloPoseAnnotator(
            model_path=args.model_path,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold
        )
        
        # 执行标注
        print(f"[Rank {RANK}] Starting pose annotation...")
        start_time = time.time()
        
        my_data = annotator.annotate_json_data(
            my_data,
            image_key=args.image_key,
            output_key=args.output_key
        )
        
        elapsed_time = time.time() - start_time
        print(f"[Rank {RANK}] Finished in {elapsed_time:.2f}s")
    else:
        print(f"[Rank {RANK}] No data to process.")
    
    # 4. 保存临时结果
    os.makedirs(args.temp_dir, exist_ok=True)
    temp_file = os.path.join(args.temp_dir, f"rank_{RANK}.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(my_data, f, ensure_ascii=False, indent=2)
    
    print(f"[Rank {RANK}] Saved temporary results to {temp_file}")
    
    # 5. 创建完成标记
    marker_dir = os.path.join(args.temp_dir, "markers")
    os.makedirs(marker_dir, exist_ok=True)
    marker_file = os.path.join(marker_dir, f"rank_{RANK}.done")
    with open(marker_file, 'w') as f:
        f.write(f"done at {time.time()}")
    
    # 6. Rank 0 等待并合并
    if RANK == 0:
        print("[Rank 0] Waiting for other processes...")
        
        max_wait_time = 3600 * 24  # 最多等待24小时
        wait_start = time.time()
        
        while True:
            done_files = glob(os.path.join(marker_dir, "*.done"))
            if len(done_files) >= WORLD_SIZE:
                break
            
            if time.time() - wait_start > max_wait_time:
                print(f"[Rank 0] Timeout waiting for processes. Only {len(done_files)}/{WORLD_SIZE} finished.")
                break
            
            time.sleep(5)
        
        print("[Rank 0] All ranks finished. Merging results...")
        
        final_data = []
        # 按顺序读取，保证原有顺序
        for r in range(WORLD_SIZE):
            t_file = os.path.join(args.temp_dir, f"rank_{r}.json")
            if os.path.exists(t_file):
                with open(t_file, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                    final_data.extend(chunk)
            else:
                print(f"[Rank 0] Warning: Missing output from Rank {r}")
        
        # 保存最终结果
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Rank 0] Final output saved to {args.output_json}")
        print(f"[Rank 0] Total processed: {len(final_data)} / {total_entries}")
        
        # 统计成功率
        success_count = sum(1 for item in final_data if item.get(args.output_key, {}).get('success', False))
        print(f"[Rank 0] Success rate: {success_count}/{len(final_data)} ({100*success_count/len(final_data):.2f}%)")
        
        # 清理临时目录
        try:
            shutil.rmtree(args.temp_dir, ignore_errors=True)
            print("[Rank 0] Cleaned up temporary directory.")
        except Exception as e:
            print(f"[Rank 0] Failed to clean temp directory: {e}")


if __name__ == "__main__":
    main()
