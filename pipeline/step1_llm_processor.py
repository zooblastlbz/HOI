import json
import os
import gc
import re
import torch
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    LLM = None
    SamplingParams = None

class LLMProcessor:
    def __init__(self, model_path, tensor_parallel_size=1):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None

    def _parse_role_section(self, caption):
        """
        从 caption 中解析 <Role> 部分，提取每个 ID 对应的人物描述。
        返回: dict, 如 {"ID_A": "A Caucasian adult male...", "ID_B": "..."}
        """
        id_descriptions = {}
        
        # 提取 <Role>...</Role> 部分
        role_match = re.search(r'<Role>(.*?)</Role>', caption, re.DOTALL)
        if not role_match:
            return id_descriptions
        
        role_text = role_match.group(1)
        
        # 匹配 <ID_X>: description 格式
        # 模式: <ID_A>: description (直到下一个 <ID_ 或结束)
        id_pattern = r'<(ID_[A-Z])>:\s*([^<]+?)(?=<ID_|$)'
        matches = re.findall(id_pattern, role_text, re.DOTALL)
        
        for id_name, description in matches:
            # 清理描述文本
            desc = description.strip().rstrip('.')
            id_descriptions[id_name] = desc
        
        return id_descriptions

    def _get_prompt(self, caption):
        """
        生成针对新 caption 格式的提示词。
        caption 包含 <Event>, <Role>, <Background>, <style>, <Scene> 等标签，
        其中人物用 <ID_A>, <ID_B> 等标记。
        """
        # 解析出人物ID和描述的映射
        id_descriptions = self._parse_role_section(caption)
        
        # 构建人物ID列表说明
        id_list_str = ""
        if id_descriptions:
            id_items = [f"- {id_name}: {desc}" for id_name, desc in id_descriptions.items()]
            id_list_str = "\n".join(id_items)
        
        return f"""# Role
You are an expert linguistic analyst for computer vision tasks. Your goal is to extract structured information from image captions to assist in visual grounding and pose estimation.

# Caption Format
The caption uses XML-like tags:
- `<Event>`: Main action/event happening in the scene. Uses <ID_A>, <ID_B>, etc. to reference specific people defined in <Role>.
- `<Role>`: Defines all person IDs with their detailed visual descriptions. Format: "<ID_X>: [full visual description]"
- `<Background>`: Environment and background description.
- `<style>`: Visual style, color scheme, and atmosphere.
- `<Scene>`: Camera angles, shot types, and spatial relationships. Also uses <ID_A>, <ID_B>, etc. to reference people.

**Important**: In sections other than <Role> (such as <Event>, <Scene>), people are referenced using ID placeholders like <ID_A>, <ID_B>. You need to look up their full descriptions in the <Role> section.

Example structure:
```
<Event><ID_A> is holding a cup while <ID_B> watches.</Event>
<Role><ID_A>: A man in black jacket and blue jeans. <ID_B>: A woman in red dress.</Role>
<Background>A coffee shop interior.</Background>
<style>Realistic style with warm lighting.</style>
<Scene><ID_A> is in the center, <ID_B> is on the right side.</Scene>
```

# Input Caption
"{caption}"

# Person ID Mapping (from <Role> section)
{id_list_str if id_list_str else "No person IDs found in <Role> section."}

# Task
Identify human body parts in the caption where the spatial location (left/right) is NOT specified (i.e., "uncertain"). For each identified part, extract the associated person, interaction object, and interaction type.

# Instructions

1. **Filter for Uncertainty**:
   - Only select body parts belonging to **humans** (referenced as <ID_A>, <ID_B>, etc.).
   - Only select body parts where the side ('left' or 'right') is **missing**.
   - *Include*: "holding a cup with hand", "kicking with leg", "reaching with arm".
   - *Exclude*: "holding a cup with left hand" (already certain), "dog wagging tail" (not human).

2. **Extract Visual Descriptions**:
   - **Person**: Use the FULL VISUAL DESCRIPTION from the <Role> section for the corresponding ID. This will be used directly for visual grounding (segmentation).
   - **Object**: Extract a CONCISE VISUAL description of the interacting object, including its relationship to the person.
     - Format: "[object features] + [relationship to person]"
     - Example: "the white cup held by the man in black jacket"
     - *Critical Rule*: If the caption implies multiple similar objects (e.g., "two chairs") and does not specify which one (e.g., "sitting on a chair"), set this to `null`. Only extract if the object is uniquely identifiable from the text.

3. **Classify Interaction**:
   Determine the nature of the interaction to guide the grounding strategy. Assign one or both categories:
   - `"close_proximity"`: Physical contact or close distance (e.g., holding, touching, kicking, carrying). Strategy: Use distance to object.
   - `"directional_orientation"`: Pointing, looking, or facing towards an object (often at a distance). Strategy: Use limb direction/vector.

# Output Format
Return a JSON object:

{{
    "uncertain_parts": [
        {{
            "person_id": "ID_A",
            "person_description": "full description from <Role> section",
            "body_part": "hand/arm/leg/foot/...",
            "interaction_object": "object description with context, or null",
            "text_span": "relevant text from caption",
            "interaction_category": ["close_proximity", "directional_orientation"]
        }}
    ]
}}

# Special Cases
- If there are people but ALL body parts already have left/right specified, return:
  {{"uncertain_parts": "all_body_parts_already_specified"}}
"""

    def load_model(self):
        if self.llm is None:
            if LLM is None:
                raise ImportError("vLLM is not installed. Cannot load local model.")
            print(f"Loading vLLM model from {self.model_path} with tensor_parallel_size={self.tensor_parallel_size}...")
            self.llm = LLM(model=self.model_path, trust_remote_code=True, tensor_parallel_size=self.tensor_parallel_size)
            print(f"Successfully loaded vLLM model.")

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
            if torch.cuda.is_available():
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
            
            # 解析并保存人物ID映射到item中，供后续步骤使用
            id_descriptions = self._parse_role_section(caption)
            item['person_id_mapping'] = id_descriptions
            
            # 如果没有人物ID，直接标记为"no human in the picture"
            if not id_descriptions:
                item['uncertain_body_part_annotation'] = "no_human_in_picture"
                continue
            
            prompt = self._get_prompt(caption)
            prompts.append(prompt)
            indices_to_process.append(i)
            
        if not prompts:
            return items

        # Load model only if there is work to do
        self.load_model()
        
        print(f"Running inference on {len(prompts)} items...")
        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
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
                    uncertain_parts = result.get("uncertain_parts", [])
                    
                    # 处理特殊情况
                    if isinstance(uncertain_parts, str):
                        # LLM 返回了字符串（如 "all_body_parts_already_specified"）
                        items[idx]['uncertain_body_part_annotation'] = uncertain_parts
                    elif isinstance(uncertain_parts, list) and len(uncertain_parts) == 0:
                        # 有人物但没有不确定的身体部位
                        items[idx]['uncertain_body_part_annotation'] = "all_body_parts_already_specified"
                    else:
                        items[idx]['uncertain_body_part_annotation'] = uncertain_parts
                else:
                    items[idx]['uncertain_body_part_annotation'] = []
            except Exception as e:
                print(f"Error parsing output for item {idx}: {e}")
                items[idx]['uncertain_body_part_annotation'] = []
        
        # Cleanup
        self.unload_model()
        return items

    def process_caption(self, caption):
        """Wrapper for single item processing."""
        item = {"caption": caption}
        self.process_batch([item])
        return {
            "uncertain_parts": item.get("uncertain_body_part_annotation", []),
            "person_id_mapping": item.get("person_id_mapping", {})
        }

if __name__ == "__main__":
    # Test code with new caption format
    processor = LLMProcessor(model_path="path/to/your/model", tensor_parallel_size=1)
    
    test_caption = '''<Event><ID_A> and <ID_B> are walking along a leaf-covered path in a forest during autumn.</Event>
<Role>There are 2 main objects in the scene. <ID_A>: A Caucasian adult male with an average build, wearing a black jacket, blue jeans, and dark shoes. <ID_B>: A Caucasian adult male with an average build, wearing a red jacket, blue jeans, and dark shoes.</Role>
<Background>The background consists of a dense forest.</Background>'''
    
    test_items = [{"caption": test_caption}]
    
    # Test parsing
    id_map = processor._parse_role_section(test_caption)
    print("Parsed ID mapping:", id_map)
    
    # Test empty case
    empty_caption = '''<Event>A beautiful sunset over the ocean.</Event>
<Role>There are no main objects in the scene.</Role>
<Background>Ocean view.</Background>'''
    
    empty_id_map = processor._parse_role_section(empty_caption)
    print("Empty ID mapping:", empty_id_map)
