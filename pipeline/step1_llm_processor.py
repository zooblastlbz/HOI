import json
import os
import gc
import torch
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    LLM = None
    # print("Warning: vLLM not installed.")

class LLMProcessor:
    def __init__(self, model_path, tensor_parallel_size=1):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None

    def _get_prompt(self, caption):
        return f"""
    # Role
    You are an expert linguistic analyst for computer vision tasks. Your goal is to extract structured information from image captions to assist in visual grounding and pose estimation.

    # Input Caption
    "{caption}"

    # Task
    Identify human body parts in the caption where the spatial location (left/right) is NOT specified (i.e., "uncertain"). For each identified part, extract the associated person, interaction object, and interaction type.

    # Instructions

    1. **Filter for Uncertainty**:
       - Only select body parts belonging to **humans**.
       - Only select body parts where the side ('left' or 'right') is **missing**.
       - *Include*: "holding a cup with hand", "kicking with leg".
       - *Exclude*: "holding a cup with left hand" (already certain), "dog wagging tail" (not human).

    2. **Extract Visual Descriptions**:
       - **Person**: Extract a CONCISE VISUAL description of the person (e.g., "man in red shirt"). This will be used for visual grounding (segmentation).
       - **Object**: Extract a CONCISE VISUAL description of the interacting object (e.g., "white cup").
         - *Critical Rule*: If the caption implies multiple similar objects (e.g., "two chairs") and does not specify which one (e.g., "sitting on a chair"), set this to `null`. Only extract if the object is uniquely identifiable from the text.

    3. **Classify Interaction**:
       Determine the nature of the interaction to guide the grounding strategy. Assign one or both categories:
       - `"close_proximity"`: Physical contact or close distance (e.g., holding, touching, kicking). Strategy: Use distance to object.
       - `"directional_orientation"`: Pointing, looking, or facing towards an object (often at a distance). Strategy: Use limb direction/vector.

    # Output Format
    Return a JSON object with a single key `"uncertain_parts"` containing a list of objects. Each object must follow this schema:

    {{
        "body_part": "string (e.g., 'hand', 'arm')",
        "person_description": "string (visual description)",
        "interaction_object": "string or null (visual description)",
        "text_span": "string (exact substring from caption)",
        "interaction_category": ["string"] (list containing 'close_proximity' and/or 'directional_orientation')
    }}

    # Example
    Caption: "A man in a black suit holding a white cup."
    Output:
    {{
        "uncertain_parts": [
            {{
                "body_part": "hand",
                "person_description": "man in black suit",
                "interaction_object": "white cup",
                "text_span": "holding a white cup",
                "interaction_category": ["close_proximity"]
            }}
        ]
    }}
    """

    def load_model(self):
        if self.llm is None:
            if LLM is None:
                raise ImportError("vLLM is not installed. Cannot load local model.")
            print(f"Loading vLLM model from {self.model_path} with tensor_parallel_size={self.tensor_parallel_size}...")
            self.llm = LLM(model=self.model_path, trust_remote_code=True, tensor_parallel_size=self.tensor_parallel_size)

    def unload_model(self):
        if self.llm is not None:
            print("Unloading vLLM model to release memory...")
            try:
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
            self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Model unloaded.")

    def process_batch(self, items):
        """
        Process a list of items in batch using vLLM.
        """
        # Identify items that need processing
        indices_to_process = []
        prompts = []
        
        for i, item in enumerate(items):
            # Skip if already annotated
            if 'uncertain_body_part_annotation' in item and item['uncertain_body_part_annotation']:
                continue
            
            caption = item.get('caption', '')
            if not caption:
                continue
                
            prompt = self._get_prompt(caption)
            prompts.append(prompt)
            indices_to_process.append(i)
            
        if not prompts:
            return items

        # Load model only if there is work to do
        self.load_model()
        
        print(f"Running inference on {len(prompts)} items...")
        sampling_params = SamplingParams(temperature=0, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        
        for idx, output in zip(indices_to_process, outputs):
            generated_text = output.outputs[0].text
            try:
                # Simple JSON extraction
                text = generated_text.strip()
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    json_str = text[start:end+1]
                    result = json.loads(json_str)
                    items[idx]['uncertain_body_part_annotation'] = result.get("uncertain_parts", [])
                else:
                    # Fallback or empty
                    items[idx]['uncertain_body_part_annotation'] = []
            except Exception as e:
                print(f"Error parsing output for item {idx}: {e}")
                items[idx]['uncertain_body_part_annotation'] = []
        
        # Cleanup
        self.unload_model()
        return items

    # Keep old method for compatibility if needed, or remove it. 
    # Since we are switching to batch, we can remove process_caption or make it use process_batch.
    def process_caption(self, caption):
        # Wrapper for single item
        item = {"caption": caption}
        self.process_batch([item])
        return {"uncertain_parts": item.get("uncertain_parts", [])}

if __name__ == "__main__":
    # Test code
    processor = LLMProcessor(model_path="path/to/your/model", tensor_parallel_size=1)
    test_items = [
        {"caption": "A man holding a cup."},
        {"caption": "A woman kicking a ball."}
    ]
    results = processor.process_batch(test_items)
    print(results)
