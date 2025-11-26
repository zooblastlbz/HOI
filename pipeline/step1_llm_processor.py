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

# Allowed Body Parts (IMPORTANT)
You can ONLY output body parts from the following list (without left/right prefix):
- nose
- eye
- ear
- shoulder
- elbow
- wrist
- hip
- knee
- ankle

Map common terms to these allowed parts:
- "hand" → "wrist"
- "arm" → "elbow" or "wrist" (depending on context)
- "leg" → "knee" or "ankle" (depending on context)
- "foot" → "ankle"
- "face" → "nose" or "eye" (depending on context)

# Instructions

1. **Filter for Uncertainty**:
   - Only select body parts belonging to **humans** (referenced as <ID_A>, <ID_B>, etc.).
   - Only select body parts where the side ('left' or 'right') is **missing**.
   - *Include*: "holding a cup with hand", "kicking with leg", "reaching with arm".
   - *Exclude*: "holding a cup with left hand" (already certain), "dog wagging tail" (not human).

2. **Extract Visual Descriptions**:
   - **Person (two versions required)**:
     - `person_description_full`: The FULL VISUAL DESCRIPTION from the <Role> section (original, complete).
     - `person_description`: A **CONCISE** version (5-8 words) with KEY DISTINCTIVE features.
       - Focus on: clothing color, unique accessories, or standout features.
       - **Good examples**: "man in black jacket", "woman with red hat", "boy in white shirt"
       - **Bad examples**: "A Caucasian adult male with an average build, wearing..." (too long)
   
   - **Object**: Extract a **CONCISE** description that links the object to the person.
     - **Format**: "[object visual features] [relation] [person's key feature]"
     - **Keep it SHORT**: Max 8-10 words.
     - **Good examples**: 
       - "red cup held by man in black jacket"
       - "umbrella near woman in red dress"
       - "basketball touched by boy in white shirt"
     - **Bad examples**: 
       - "cup" (no person link, not distinctive)
       - "the white cup that is being held by the man wearing a black jacket and blue jeans" (too long)
     - *Critical Rule*: If the object cannot be uniquely identified (e.g., "one of two chairs"), set to `null`.

3. **Classify Interaction**:
   Determine the nature of the interaction to guide the grounding strategy. Assign one or both categories:
   - `"close_proximity"`: Physical contact or close distance (e.g., holding, touching, kicking, carrying). Strategy: Use distance to object.
   - `"directional_orientation"`: Pointing, looking, or facing towards an object (often at a distance). Strategy: Use limb direction/vector.

# Output Format
**IMPORTANT: You MUST output ONLY valid JSON. Do not include any explanation, markdown formatting, or text outside the JSON object.**

Output a single JSON object with the following structure:
```json
{{
    "uncertain_parts": [
        {{
            "person_id": "ID_A",
            "person_description_full": "complete original description from <Role> section",
            "person_description": "concise 5-8 word description for grounding",
            "body_part": "one of: nose/eye/ear/shoulder/elbow/wrist/hip/knee/ankle",
            "interaction_object": "concise description linking object to person, or null",
            "text_span": "relevant text from caption",
            "interaction_category": ["close_proximity", "directional_orientation"]
        }}
    ]
}}
```

# Special Cases
- If there are people but ALL body parts already have left/right specified, return:
  ```json
  {{"uncertain_parts": "all_body_parts_already_specified"}}
  ```
- If no uncertain body parts are found, return:
  ```json
  {{"uncertain_parts": []}}
  ```

# Response
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

    def _extract_json(self, text):
        """
        从 LLM 输出中提取 JSON 对象。
        支持多种格式：纯 JSON、markdown 代码块、带有额外文本等。
        """
        text = text.strip()
        
        # 尝试方法1：直接解析（纯 JSON 输出）
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试方法2：提取 ```json ... ``` 代码块
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试方法3：找到第一个 { 和最后一个 } 之间的内容
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                json_str = text[start:end+1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 尝试方法4：逐行查找，处理可能的多个 JSON 对象
        lines = text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            if '{' in line and not in_json:
                in_json = True
            if in_json:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    try:
                        json_str = '\n'.join(json_lines)
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        json_lines = []
                        in_json = False
                        brace_count = 0
        
        # 所有方法都失败
        return None

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
                # 使用增强的 JSON 提取方法
                result = self._extract_json(generated_text)
                
                if result is None:
                    print(f"Warning: Failed to extract JSON for item {idx}")
                    print(f"Raw output: {generated_text[:200]}...")
                    items[idx]['uncertain_body_part_annotation'] = []
                    items[idx]['llm_parse_error'] = "failed_to_extract_json"
                    continue
                
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
                    
            except Exception as e:
                print(f"Error parsing output for item {idx}: {e}")
                print(f"Raw output: {generated_text}...")
                items[idx]['uncertain_body_part_annotation'] = []
                items[idx]['llm_parse_error'] = str(e)
        
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
