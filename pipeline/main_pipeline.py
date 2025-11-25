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
    def __init__(self, llm_model_path=None):
        self.llm = LLMProcessor(model_path=llm_model_path)
        self.grounding = GroundingModule()
        self.pose = PoseModule()

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

    def process_step2_grounding(self, item):
        image_path = item.get('image_path')
        # If image_path is missing or invalid, we might skip grounding but keep the item
        if not image_path or not os.path.exists(image_path):
            # print(f"Warning: Image path not found: {image_path}")
            return item
            
        annotations = item.get('uncertain_body_part_annotation', [])
        for part in annotations:
            person_desc = part.get("person_description")
            obj_name = part.get("interaction_object")
            
            # Ground Person
            if 'person_bbox' not in part:
                person_bbox = self.grounding.detect_person(image_path, person_desc)
                part['person_bbox'] = person_bbox
            
            # Ground Object
            if obj_name and 'object_bbox' not in part:
                person_bbox = part.get('person_bbox')
                obj_bbox = self.grounding.detect_object_in_roi(image_path, obj_name, person_bbox)
                part['object_bbox'] = obj_bbox
        
        item['uncertain_body_part_annotation'] = annotations
        return item

    def process_step3_pose(self, item):
        image_path = item.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return item
            
        annotations = item.get('uncertain_body_part_annotation', [])
        for part in annotations:
            person_bbox = part.get('person_bbox')
            obj_bbox = part.get('object_bbox')
            interaction_types = part.get("interaction_category", [])
            body_part = part.get("body_part")
            
            if not person_bbox:
                part["resolved_side"] = "unknown (no person bbox)"
                continue
                
            # Pose Estimation
            keypoints = self.pose.get_keypoints(image_path, person_bbox)
            # part['keypoints'] = keypoints # Optional: save keypoints to output
            
            final_side = "uncertain"
            
            # Strategy A: Close Proximity (Distance-based)
            if "close_proximity" in interaction_types and obj_bbox:
                if "hand" in body_part or "arm" in body_part:
                    final_side = self.resolve_side(keypoints, obj_bbox)
            
            # Strategy B: Directional Orientation (Vector-based)
            elif "directional_orientation" in interaction_types:
                final_side = "needs_directional_logic" # Placeholder for vector logic

            part["resolved_side"] = final_side
            
        item['uncertain_body_part_annotation'] = annotations
        return item

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
            self.run_step("Step 2: Visual Grounding", self.process_step2_grounding, in_s2, step2_out, num_samples=num_samples)
            
        if step is None or step == 3:
            self.run_step("Step 3: Pose & Resolution", self.process_step3_pose, in_s3, step3_out, num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.json", help="Input data file or LMDB directory")
    parser.add_argument("--output_dir", default="results", help="Directory to save intermediate and final results")
    parser.add_argument("--llm_model", default="facebook/opt-125m", help="Path to local LLM model for vLLM")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process from LMDB")
    parser.add_argument("--step", type=int, default=None, choices=[1, 2, 3], help="Run a specific step (1, 2, or 3). If not specified, runs all steps.")
    args = parser.parse_args()
    
    pipeline = HOIPipeline(llm_model_path=args.llm_model)
    pipeline.run_pipeline(args.input, args.output_dir, num_samples=args.num_samples, step=args.step)
