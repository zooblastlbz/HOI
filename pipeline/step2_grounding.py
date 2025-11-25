import os
import torch
import numpy as np
from PIL import Image

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

    def _get_bbox_from_mask(self, mask):
        """
        Convert a binary mask (numpy or tensor) to [x1, y1, x2, y2].
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # mask shape: (H, W)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def detect_person(self, image_path, person_description):
        """
        Step 2.1: Detect the specific person described by the text using SAM3.
        Returns: Bounding Box [x1, y1, x2, y2] of the person.
        """
        if not self.model:
            print(f"[Mock] SAM3 Detecting person: '{person_description}'")
            return [100, 100, 400, 400]

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
            
            # results[0] corresponds to the first (and only) image
            masks = results[0]['masks'] # Shape: (N, H, W)
            scores = results[0].get('scores', [])
            
            if len(masks) > 0:
                # If multiple matches, pick the one with highest score if available, else first
                best_idx = 0
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                
                best_mask = masks[best_idx]
                bbox = self._get_bbox_from_mask(best_mask)
                return bbox if bbox else None
            else:
                print(f"No person found for description: {person_description}")
                return None

        except Exception as e:
            print(f"Error in detect_person: {e}")
            return None

    def detect_person_batch(self, items):
        """
        Batch detection for persons.
        items: List of dicts with 'image_path' and 'person_description'
        Returns: List of bounding boxes
        """
        if not self.model:
            print(f"[Mock] SAM3 Batch detecting {len(items)} persons")
            return [[100, 100, 400, 400] for _ in items]

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
                        masks = post_results[k]['masks']
                        scores = post_results[k].get('scores', [])
                        
                        if len(masks) > 0:
                            best_idx = 0
                            if len(scores) > 0:
                                best_idx = torch.argmax(scores).item()
                            best_mask = masks[best_idx]
                            bbox = self._get_bbox_from_mask(best_mask)
                            batch_results[idx] = bbox
                            
                except Exception as e:
                    print(f"Error in batch person detection: {e}")
            
            results_list.extend(batch_results)
        
        return results_list

    def detect_object_in_roi(self, image_path, object_name, roi_bbox):
        """
        Step 2.2: Detect the interaction object within the Person's ROI using SAM3.
        Returns: Bounding Box [x1, y1, x2, y2] of the object.
        """
        if not self.model:
            print(f"[Mock] SAM3 Detecting object: '{object_name}' inside ROI {roi_bbox}")
            return [150, 150, 200, 200]

        print(f"SAM3 processing object: '{object_name}' in ROI")
        
        try:
            full_image = Image.open(image_path).convert("RGB")
            
            # Crop image to ROI
            x1, y1, x2, y2 = map(int, roi_bbox)
            w, h = full_image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                print("Invalid ROI for object detection.")
                return None
                
            crop_image = full_image.crop((x1, y1, x2, y2))
            
            # SAM3 inference on crop
            inputs = self.processor(images=[crop_image], text=[object_name], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            masks = results[0]['masks']
            scores = results[0].get('scores', [])
            
            if len(masks) > 0:
                best_idx = 0
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    
                best_mask = masks[best_idx]
                local_bbox = self._get_bbox_from_mask(best_mask)
                
                if local_bbox:
                    # Convert local crop coordinates back to global coordinates
                    lx1, ly1, lx2, ly2 = local_bbox
                    return [lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1]
            
            return None

        except Exception as e:
            print(f"Error in detect_object_in_roi: {e}")
            return None

    def detect_object_in_roi_batch(self, items):
        """
        Batch detection for objects in ROI.
        items: List of dicts with 'image_path', 'object_name', 'roi_bbox'
        Returns: List of bounding boxes
        """
        if not self.model:
            print(f"[Mock] SAM3 Batch detecting {len(items)} objects")
            return [[150, 150, 200, 200] for _ in items]

        results_list = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            crop_images = []
            texts = []
            offsets = []  # Store (x1, y1) for coordinate conversion
            
            for item in batch:
                try:
                    roi_bbox = item.get('roi_bbox')
                    if not roi_bbox:
                        crop_images.append(None)
                        texts.append("")
                        offsets.append((0, 0))
                        continue
                        
                    full_image = Image.open(item['image_path']).convert("RGB")
                    x1, y1, x2, y2 = map(int, roi_bbox)
                    w, h = full_image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        crop_images.append(None)
                        texts.append("")
                        offsets.append((0, 0))
                        continue
                    
                    crop_image = full_image.crop((x1, y1, x2, y2))
                    crop_images.append(crop_image)
                    texts.append(item['object_name'])
                    offsets.append((x1, y1))
                    
                except Exception as e:
                    print(f"Error loading image {item.get('image_path')}: {e}")
                    crop_images.append(None)
                    texts.append("")
                    offsets.append((0, 0))
            
            # Filter valid items
            valid_indices = [j for j, img in enumerate(crop_images) if img is not None]
            valid_images = [crop_images[j] for j in valid_indices]
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
                        masks = post_results[k]['masks']
                        scores = post_results[k].get('scores', [])
                        offset_x, offset_y = offsets[idx]
                        
                        if len(masks) > 0:
                            best_idx = 0
                            if len(scores) > 0:
                                best_idx = torch.argmax(scores).item()
                            best_mask = masks[best_idx]
                            local_bbox = self._get_bbox_from_mask(best_mask)
                            
                            if local_bbox:
                                lx1, ly1, lx2, ly2 = local_bbox
                                batch_results[idx] = [lx1 + offset_x, ly1 + offset_y, lx2 + offset_x, ly2 + offset_y]
                            
                except Exception as e:
                    print(f"Error in batch object detection: {e}")
            
            results_list.extend(batch_results)
        
        return results_list
