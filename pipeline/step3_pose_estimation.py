import cv2
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PoseModule:
    def __init__(self, model_path="yolo11n-pose.pt"):
        if YOLO:
            print(f"Loading YOLO11-Pose model from {model_path}...")
            # 自动下载或加载本地模型
            self.model = YOLO(model_path)
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
