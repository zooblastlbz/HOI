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

    def detect_person(self, image_path, person_description):
        """
        Step 2.1: Detect the specific person described by the text using SAM3.
        Returns: 
            - Bounding Box [x1, y1, x2, y2] if exactly ONE person is detected
            - None if no person or multiple persons detected (invalid data)
        """
        if not self.model:
            raise ValueError("SAM3 model not loaded. Cannot perform person detection.")

        print(f"SAM3 processing: '{person_description}' in {image_path}")
        
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
        Batch detection for persons.
        items: List of dicts with 'image_path' and 'person_description'
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
                    texts.append(item['person_description'])
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
        items: List of dicts with 'image_path', 'object_name'
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
                    texts.append(item['object_name'])
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

    # 保留旧方法名作为别名，向后兼容
    def detect_object_in_roi(self, image_path, object_name, roi_bbox=None):
        """Deprecated: Use detect_object instead. ROI is now ignored."""
        return self.detect_object(image_path, object_name)

    def detect_object_in_roi_batch(self, items):
        """Deprecated: Use detect_object_batch instead. ROI is now ignored."""
        # 转换参数格式
        new_items = [{'image_path': item['image_path'], 'object_name': item['object_name']} for item in items]
        return self.detect_object_batch(new_items)

    def process_items_batch(self, items):
        """
        批量处理完整的 grounding 流程。
        
        Args:
            items: 列表，每个 item 包含:
                - image_path: 图片路径
                - uncertain_body_part_annotation: 不确定身体部位的列表或字符串
                
        Returns:
            处理后的 items 列表
        """
        # 第一步：过滤，分离需要处理和不需要处理的 items
        items_to_process = []  # (index, annotations)
        
        for i, item in enumerate(items):
            annotations = item.get('uncertain_body_part_annotation', [])
            
            # 过滤：跳过不需要处理的情况
            if isinstance(annotations, str):
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = annotations
                continue
            
            if not annotations or len(annotations) == 0:
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = "no_uncertain_body_parts"
                continue
            
            image_path = item.get('image_path')
            if not image_path or not os.path.exists(image_path):
                item['grounding_skipped'] = True
                item['grounding_skip_reason'] = "image_not_found"
                continue
            
            # 有不确定的身体部位，需要处理
            item['grounding_skipped'] = False
            items_to_process.append((i, annotations))
        
        if not items_to_process:
            print("No items need grounding processing.")
            return items
        
        print(f"Grounding: {len(items_to_process)} items need processing (out of {len(items)} total)")
        
        # 第二步：收集所有需要检测人物的任务
        person_tasks = []
        task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, annotations in items_to_process:
            image_path = items[item_idx].get('image_path')
            for part_idx, part in enumerate(annotations):
                if 'person_bbox' not in part:
                    person_tasks.append({
                        'image_path': image_path,
                        'person_description': part.get('person_description', '')
                    })
                    task_mapping.append((item_idx, part_idx))
        
        # 第三步：批量检测人物
        if person_tasks:
            print(f"Batch detecting {len(person_tasks)} persons...")
            person_results = self.detect_person_batch(person_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), bbox in zip(task_mapping, person_results):
                items[item_idx]['uncertain_body_part_annotation'][part_idx]['person_bbox'] = bbox
        
        # 第四步：收集所有需要检测物品的任务
        object_tasks = []
        obj_task_mapping = []  # (item_idx, part_idx)
        
        for item_idx, _ in items_to_process:
            image_path = items[item_idx].get('image_path')
            annotations = items[item_idx]['uncertain_body_part_annotation']
            
            for part_idx, part in enumerate(annotations):
                person_bbox = part.get('person_bbox')
                obj_name = part.get('interaction_object')
                
                # 只有人物检测成功且有交互物品时才检测物品
                if person_bbox and obj_name and 'object_bboxes' not in part:
                    object_tasks.append({
                        'image_path': image_path,
                        'object_name': obj_name,
                        'roi_bbox': person_bbox
                    })
                    obj_task_mapping.append((item_idx, part_idx))
        
        # 第五步：批量检测物品
        if object_tasks:
            print(f"Batch detecting {len(object_tasks)} objects...")
            object_results = self.detect_object_in_roi_batch(object_tasks)
            
            # 将结果写回
            for (item_idx, part_idx), obj_bboxes in zip(obj_task_mapping, object_results):
                part = items[item_idx]['uncertain_body_part_annotation'][part_idx]
                part['object_bboxes'] = obj_bboxes
                part['object_bbox'] = obj_bboxes[0] if obj_bboxes else None
        
        # 第六步：标记 grounding 有效性
        for item_idx, _ in items_to_process:
            annotations = items[item_idx]['uncertain_body_part_annotation']
            for part in annotations:
                if part.get('person_bbox') is None:
                    part['grounding_valid'] = False
                    part['grounding_error'] = "Invalid: person detection failed (0 or multiple persons)"
                else:
                    part['grounding_valid'] = True
        
        return items
