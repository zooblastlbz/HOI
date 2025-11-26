import json
import math
import os
import sys
import argparse
import pickle
try:
    import lmdb
except ImportError:
    lmdb = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step1_llm_processor import LLMProcessor
from step2_grounding import GroundingModule
from step3_pose_estimation import PoseModule

class HOIPipeline:
    def __init__(self, llm_model_path=None, 
                 llm_tensor_parallel_size=1,
                 sam_model_path="facebook/sam3-demo",
                 sam_gpu_ids=None, sam_batch_size=1,
                 yolo_model_path="yolo11n-pose.pt",
                 yolo_gpu_ids=None, yolo_batch_size=1):
        """
        初始化 HOI Pipeline (延迟加载模式)
        
        Args:
            llm_model_path: vLLM模型路径
            llm_tensor_parallel_size: vLLM的tensor parallel size (GPU数量)
            sam_model_path: SAM3模型路径或HuggingFace ID
            sam_gpu_ids: SAM使用的GPU ID列表，如 [0, 1, 2]
            sam_batch_size: SAM的batch size
            yolo_model_path: YOLO模型路径
            yolo_gpu_ids: YOLO使用的GPU ID列表，如 [0, 1]
            yolo_batch_size: YOLO的batch size
        """
        # 保存配置，延迟加载模型
        self.llm_model_path = llm_model_path
        self.llm_tensor_parallel_size = llm_tensor_parallel_size
        self.sam_model_path = sam_model_path
        self.sam_gpu_ids = sam_gpu_ids
        self.sam_batch_size = sam_batch_size
        self.yolo_model_path = yolo_model_path
        self.yolo_gpu_ids = yolo_gpu_ids
        self.yolo_batch_size = yolo_batch_size
        
        # 模型实例，延迟初始化
        self._llm = None
        self._grounding = None
        self._pose = None

    @property
    def llm(self):
        """延迟加载 LLM 模块"""
        if self._llm is None:
            print("Initializing LLM module...")
            self._llm = LLMProcessor(
                model_path=self.llm_model_path, 
                tensor_parallel_size=self.llm_tensor_parallel_size
            )
        return self._llm

    @property
    def grounding(self):
        """延迟加载 Grounding 模块"""
        if self._grounding is None:
            print("Initializing Grounding (SAM) module...")
            self._grounding = GroundingModule(
                model_name=self.sam_model_path, 
                gpu_ids=self.sam_gpu_ids, 
                batch_size=self.sam_batch_size
            )
        return self._grounding

    @property
    def pose(self):
        """延迟加载 Pose 模块"""
        if self._pose is None:
            print("Initializing Pose (YOLO) module...")
            self._pose = PoseModule(
                model_path=self.yolo_model_path, 
                gpu_ids=self.yolo_gpu_ids, 
                batch_size=self.yolo_batch_size
            )
        return self._pose

    def load_lmdb_data(self, lmdb_path, num_samples=None):
        if lmdb is None:
            print("Error: lmdb module not installed.")
            return []
            
        print(f"Loading data from LMDB: {lmdb_path}")
        data_list = []
        if not os.path.exists(lmdb_path):
            print(f"Error: LMDB path {lmdb_path} not found.")
            return []

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            count = 0
            total_entries = txn.stat()['entries']
            # Iterate based on integer keys starting from 0
            # If num_samples is None, we try to load all entries (assuming keys 0..total_entries-1)
            limit = num_samples if num_samples is not None else total_entries
            
            # We iterate up to total_entries + buffer to be safe, or until num_samples is reached
            max_iter = total_entries + 1000
            
            for i in range(max_iter):
                if num_samples is not None and count >= num_samples:
                    break
                
                key = f'{i}'.encode()
                value_bytes = txn.get(key)
                
                if value_bytes:
                    try:
                        data = pickle.loads(value_bytes)
                    except Exception as e:
                        print(f"Error loading pickle for key {key}: {e}")
                        continue

                    # --- 1. 获取图片路径 ---
                    image_path = None
                    
                    # 获取原始路径列表
                    f_list = data.get('frame_dir_list', [])
                    if isinstance(f_list, str): f_list = [f_list]
                    
                    if not f_list:
                        # print("Skipping: No frame_dir_list found.")
                        continue

                    raw_path = f_list[0]
                    
                    # 路径判断逻辑
                    if os.path.isfile(raw_path):
                        image_path = raw_path
                    else:
                        # 尝试拼接第一帧
                        potential_path = os.path.join(raw_path, '000001.jpg')
                        if os.path.exists(potential_path):
                            image_path = potential_path
                        else:
                            print(f"Warning: File not found locally: {raw_path}")
                            image_path = raw_path 
                
                    # Caption
                    caption = data.get('caption', '')
                    if not caption:
                         caption = data.get('text', '')

                    data_list.append({
                        "id": str(i),
                        "image_path": image_path,
                        "caption": caption
                    })
                    count += 1
                else:
                    # If key not found and we are expecting sequential keys, 
                    # we might have reached the end if num_samples was None.
                    if num_samples is None and i >= total_entries:
                        break
                
        env.close()
        print(f"Loaded {len(data_list)} items from LMDB.")
        return data_list

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def resolve_side(self, keypoints, object_bbox):
        """
        Determine if it's left or right hand based on distance to object center.
        """
        if not object_bbox:
            return "unknown"
        
        # Calculate object center
        obj_center = [
            (object_bbox[0] + object_bbox[2]) / 2,
            (object_bbox[1] + object_bbox[3]) / 2
        ]

        left_wrist = keypoints.get("left_wrist")
        right_wrist = keypoints.get("right_wrist")

        if not left_wrist or not right_wrist:
            return "unknown (missing keypoints)"

        dist_left = self.calculate_distance(left_wrist, obj_center)
        dist_right = self.calculate_distance(right_wrist, obj_center)

        if dist_left < dist_right:
            return "left"
        else:
            return "right"

    def run_step(self, step_name, process_func, input_file, output_file, batch_mode=False, num_samples=None):
        print(f"\n=== Running {step_name} ===")
        
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found.")
            return

        # Determine input type
        data = []
        if os.path.isdir(input_file):
            # Assume LMDB if directory
            data = self.load_lmdb_data(input_file, num_samples)
        else:
            # Assume JSON
            with open(input_file, 'r') as f:
                data = json.load(f)
                if num_samples is not None:
                    data = data[:num_samples]
            
        # Resume logic
        processed_data = []
        processed_ids = set()
        
        if os.path.exists(output_file):
            print(f"Found existing output {output_file}, checking for resume...")
            with open(output_file, 'r') as f:
                try:
                    processed_data = json.load(f)
                    for item in processed_data:
                        if 'id' in item:
                            processed_ids.add(item['id'])
                except json.JSONDecodeError:
                    print("Output file corrupted, starting fresh.")
                    processed_data = []
        
        items_to_process = []
        for item in data:
            # If item has no ID, we can't reliably resume, so we might re-process or skip.
            # Assuming items have 'id'.
            if 'id' in item and item['id'] in processed_ids:
                continue
            items_to_process.append(item)
            
        if not items_to_process:
            print(f"All items already processed for {step_name}.")
            return

        print(f"Processing {len(items_to_process)} items...")
        
        new_results = []
        
        if batch_mode:
            try:
                # Process all items at once
                new_results = process_func(items_to_process)
            except Exception as e:
                print(f"Error in batch processing for {step_name}: {e}")
                # In case of error, we might return empty or original items
                new_results = items_to_process
        else:
            # Use tqdm if available
            iterator = tqdm(items_to_process, desc=step_name)
            
            for item in iterator:
                try:
                    # Process the item
                    processed_item = process_func(item)
                    new_results.append(processed_item)
                except Exception as e:
                    print(f"Error processing item {item.get('id', 'unknown')}: {e}")
                    item['error'] = str(e)
                    new_results.append(item)
                
        # Combine and Save
        final_data = processed_data + new_results
        # Sort by ID if possible to keep order
        # final_data.sort(key=lambda x: x.get('id', 0)) 
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_file}")

    def process_step1_llm(self, items):
        # Batch processing for LLM
        return self.llm.process_batch(items)

    def process_step2_grounding(self, items):
        # Batch processing for Grounding
        return self.grounding.process_items_batch(items)

    def process_step3_pose(self, items):
        # Batch processing for Pose Estimation
        return self.pose.process_items_batch(items)

    def check_input_compatibility(self, file_path, step):
        if not os.path.exists(file_path):
            return False
        if os.path.isdir(file_path):
            return step == 1
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list) or not data:
                    return False
                item = data[0]
                if step == 2:
                    return 'uncertain_body_part_annotation' in item
                if step == 3:
                    annotations = item.get('uncertain_body_part_annotation', [])
                    if not annotations: return True
                    return 'person_bbox' in annotations[0]
        except:
            return False
        return True

    def run_pipeline(self, input_file, output_dir, num_samples=None, step=None):
        step1_out = os.path.join(output_dir, "step1_llm.json")
        step2_out = os.path.join(output_dir, "step2_grounding.json")
        step3_out = os.path.join(output_dir, "step3_final.json")
        
        in_s1 = input_file
        
        # Smart input detection for Step 2
        in_s2 = step1_out
        if step == 2:
            in_s2 = input_file
            if not self.check_input_compatibility(in_s2, 2) and os.path.exists(step1_out):
                print(f"Input {in_s2} not compatible with Step 2. Auto-switching to {step1_out}")
                in_s2 = step1_out
                
        # Smart input detection for Step 3
        in_s3 = step2_out
        if step == 3:
            in_s3 = input_file
            if not self.check_input_compatibility(in_s3, 3) and os.path.exists(step2_out):
                print(f"Input {in_s3} not compatible with Step 3. Auto-switching to {step2_out}")
                in_s3 = step2_out

        if step is None or step == 1:
            self.run_step("Step 1: LLM Analysis", self.process_step1_llm, in_s1, step1_out, batch_mode=True, num_samples=num_samples)
        
        if step is None or step == 2:
            self.run_step("Step 2: Visual Grounding", self.process_step2_grounding, in_s2, step2_out, batch_mode=True, num_samples=num_samples)
            
        if step is None or step == 3:
            self.run_step("Step 3: Pose & Resolution", self.process_step3_pose, in_s3, step3_out, batch_mode=True, num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.json", help="Input data file or LMDB directory")
    parser.add_argument("--output_dir", default="results", help="Directory to save intermediate and final results")
    parser.add_argument("--llm_model", default="facebook/opt-125m", help="Path to local LLM model for vLLM")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process from LMDB")
    parser.add_argument("--step", type=int, default=None, choices=[1, 2, 3], help="Run a specific step (1, 2, or 3). If not specified, runs all steps.")
    
    # 模型路径配置
    parser.add_argument("--sam_model", default="facebook/sam3-demo", 
                        help="Path to SAM3 model or HuggingFace model ID")
    parser.add_argument("--yolo_model", default="yolo11n-pose.pt", 
                        help="Path to YOLO pose model")
    
    # GPU 配置参数
    parser.add_argument("--llm_tensor_parallel_size", type=int, default=1, 
                        help="Tensor parallel size for vLLM (number of GPUs for LLM)")
    parser.add_argument("--sam_gpu_ids", type=str, default="0", 
                        help="Comma-separated GPU IDs for SAM model (e.g., '0,1,2')")
    parser.add_argument("--sam_batch_size", type=int, default=1, 
                        help="Batch size for SAM model")
    parser.add_argument("--yolo_gpu_ids", type=str, default="0", 
                        help="Comma-separated GPU IDs for YOLO model (e.g., '0,1')")
    parser.add_argument("--yolo_batch_size", type=int, default=1, 
                        help="Batch size for YOLO model")
    
    args = parser.parse_args()
    
    # 解析 GPU ID 列表
    sam_gpu_ids = [int(x.strip()) for x in args.sam_gpu_ids.split(',') if x.strip()]
    yolo_gpu_ids = [int(x.strip()) for x in args.yolo_gpu_ids.split(',') if x.strip()]
    
    pipeline = HOIPipeline(
        llm_model_path=args.llm_model,
        llm_tensor_parallel_size=args.llm_tensor_parallel_size,
        sam_model_path=args.sam_model,
        sam_gpu_ids=sam_gpu_ids,
        sam_batch_size=args.sam_batch_size,
        yolo_model_path=args.yolo_model,
        yolo_gpu_ids=yolo_gpu_ids,
        yolo_batch_size=args.yolo_batch_size
    )
    pipeline.run_pipeline(args.input, args.output_dir, num_samples=args.num_samples, step=args.step)
