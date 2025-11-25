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
        
        if YOLO:
            print(f"Loading YOLO11-Pose model from {model_path} on {self.num_gpus} GPU(s): {self.gpu_ids}, batch_size={self.batch_size}...")
            # 为每个GPU加载一个模型实例
            for gpu_id in self.gpu_ids:
                model = YOLO(model_path)
                model.to(f"cuda:{gpu_id}")
                self.models.append(model)
            # 主模型用于单个推理
            self.model = self.models[0] if self.models else None
        else:
            print("Warning: 'ultralytics' package not installed. Pose estimation will fail.")
            self.model = None

    def get_keypoints(self, image_path, roi_bbox):
        """
        Step 3: Detect human keypoints within the Person's ROI using YOLO11-Pose.
        Returns: Dictionary of keypoints { "left_wrist": [x, y], "right_wrist": [x, y], ... }
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
        results = self.model(roi_img, verbose=False)
        
        keypoints_dict = {}
        
        # YOLO-Pose Keypoint Mapping (COCO format)
        # 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
        # 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
        # 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
        # 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        
        if results and results[0].keypoints is not None:
            # 取置信度最高的一个人（假设 ROI 里主要是目标人物）
            # keypoints.xy shape: (N, 17, 2)
            kpts = results[0].keypoints.xy.cpu().numpy()
            
            if len(kpts) > 0:
                # 这里简单取第一个检测到的人，实际可以根据中心点距离优化
                person_kpts = kpts[0] 
                
                # 转换回原图坐标
                def to_global(kp_local):
                    if kp_local[0] == 0 and kp_local[1] == 0: return None # 未检测到
                    return [float(kp_local[0] + x1), float(kp_local[1] + y1)]

                keypoints_dict["nose"] = to_global(person_kpts[0])
                keypoints_dict["left_wrist"] = to_global(person_kpts[9])
                keypoints_dict["right_wrist"] = to_global(person_kpts[10])
                keypoints_dict["left_shoulder"] = to_global(person_kpts[5])
                keypoints_dict["right_shoulder"] = to_global(person_kpts[6])
                
        return keypoints_dict

    def get_keypoints_batch(self, items):
        """
        Batch detection for keypoints.
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
                    # 使用第一个模型进行批量推理
                    # 根据GPU数量分配任务
                    if self.num_gpus > 1:
                        # 多GPU并行处理
                        import concurrent.futures
                        
                        def process_on_gpu(gpu_idx, images):
                            model = self.models[gpu_idx]
                            return model(images, verbose=False)
                        
                        # 分配图像到不同GPU
                        images_per_gpu = [[] for _ in range(self.num_gpus)]
                        indices_per_gpu = [[] for _ in range(self.num_gpus)]
                        
                        for k, img in enumerate(roi_images):
                            gpu_idx = k % self.num_gpus
                            images_per_gpu[gpu_idx].append(img)
                            indices_per_gpu[gpu_idx].append(k)
                        
                        all_results = [None] * len(roi_images)
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                            futures = {}
                            for gpu_idx in range(self.num_gpus):
                                if images_per_gpu[gpu_idx]:
                                    futures[executor.submit(process_on_gpu, gpu_idx, images_per_gpu[gpu_idx])] = gpu_idx
                            
                            for future in concurrent.futures.as_completed(futures):
                                gpu_idx = futures[future]
                                try:
                                    gpu_results = future.result()
                                    for k, result in enumerate(gpu_results):
                                        original_idx = indices_per_gpu[gpu_idx][k]
                                        all_results[original_idx] = result
                                except Exception as e:
                                    print(f"Error in GPU {gpu_idx} processing: {e}")
                        
                        results = all_results
                    else:
                        # 单GPU批量推理
                        results = self.model(roi_images, verbose=False)
                    
                    for k, (result, valid_idx) in enumerate(zip(results, valid_indices)):
                        if result is None:
                            continue
                            
                        x1, y1 = offsets[k]
                        keypoints_dict = {}
                        
                        if result.keypoints is not None:
                            kpts = result.keypoints.xy.cpu().numpy()
                            
                            if len(kpts) > 0:
                                person_kpts = kpts[0]
                                
                                def to_global(kp_local):
                                    if kp_local[0] == 0 and kp_local[1] == 0:
                                        return None
                                    return [float(kp_local[0] + x1), float(kp_local[1] + y1)]
                                
                                keypoints_dict["nose"] = to_global(person_kpts[0])
                                keypoints_dict["left_wrist"] = to_global(person_kpts[9])
                                keypoints_dict["right_wrist"] = to_global(person_kpts[10])
                                keypoints_dict["left_shoulder"] = to_global(person_kpts[5])
                                keypoints_dict["right_shoulder"] = to_global(person_kpts[6])
                        
                        batch_results[valid_idx] = keypoints_dict
                        
                except Exception as e:
                    print(f"Error in batch pose estimation: {e}")
            
            results_list.extend(batch_results)
        
        return results_list
