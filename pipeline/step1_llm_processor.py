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
You are an expert linguistic analyst for computer vision tasks. Your goal is to extract all body parts involved in actions/interactions from image captions, normalizing laterality information.

# Caption Format
The caption uses XML-like tags:
- `<Event>`: Main action/event happening in the scene. Uses <ID_A>, <ID_B>, etc. to reference specific people defined in <Role>.
- `<Role>`: Defines all person IDs with their detailed visual descriptions. Format: "<ID_X>: [full visual description]"
- `<Background>`: Environment and background description.
- `<style>`: Visual style, color scheme, and atmosphere.
- `<Scene>`: Camera angles, shot types, and spatial relationships. Also uses <ID_A>, <ID_B>, etc. to reference people.

**Important**: In sections other than <Role> (such as <Event>, <Scene>), people are referenced using ID placeholders like <ID_A>, <ID_B>. You need to look up their full descriptions in the <Role> section.

# Input Caption
"{caption}"

# Person ID Mapping (from <Role> section)
{id_list_str if id_list_str else "No person IDs found in <Role> section."}

# Task
Analyze the caption and extract ALL human body parts involved in actions or interactions.
For each body part:
1. Extract the interaction details (object, text span, interaction type, person)
2. **NORMALIZE the body part name by REMOVING any laterality (left/right) information**
3. Record whether laterality was originally specified in the caption
4. **Include the FULL person description from <Role> section**
5. **Include a BRIEF person description for grounding**

# Laterality Normalization Rules
**ALWAYS normalize body part names by removing left/right:**
- "left hand" → body_part: "hand of ...", laterality_specified: true
- "right foot" → body_part: "foot of ...", laterality_specified: true  
- "left arm" → body_part: "arm of ...", laterality_specified: true
- "right leg" → body_part: "leg of ...", laterality_specified: true
- "hand" (unspecified) → body_part: "hand of ...", laterality_specified: false
- "both hands" → DO NOT extract (see exclusion rules below)

# What NOT to Extract (Exclusion Rules)
1. **Both hands/feet doing the SAME action on the SAME object**: 
   - "holding a box with both hands" → DO NOT extract
   - "carrying a tray with two hands" → DO NOT extract
   - "hands gripping the steering wheel" → DO NOT extract
2. **Symmetric actions where left/right doesn't matter**:
   - "clapping hands" → DO NOT extract
   - "folding arms" → DO NOT extract
3. **Non-lateral body parts**: "head", "face", "back", "chest", "torso" → DO NOT extract (no left/right distinction)

# What TO Extract
1. **Any single body part action (with or without laterality specified)**:
   - "holding a cup" → extract as "hand of ..."
   - "left hand holding a cup" → extract as "hand of ..." (laterality_specified: true)
   - "right foot kicking the ball" → extract as "foot of ..." (laterality_specified: true)
2. **Two body parts doing DIFFERENT actions or with DIFFERENT objects → Extract as TWO separate entries**:
   - "left hand holds phone, right hand holds cup" → extract TWO entries
   - "holds a phone and a cup" → extract TWO entries
   - "left hand waving, right hand holding bag" → extract TWO entries (different actions)
   - "kicking with left foot while balancing on right foot" → extract TWO entries

# Annotation Rules (ALL descriptions must be ≤32 tokens)

## 1. body_part
- **Format: "[body_part] of [person_brief_description]"**
- The body part name should be NORMALIZED (no left/right): "hand", "foot", "arm", "leg", "finger", "knee", "elbow", "shoulder"
- Append " of [person_brief_description]" to identify which person
- Example: "hand of man in blue shirt", "foot of woman in red dress"
- **NEVER include "left", "right", or "both" in this field**

## 2. laterality_specified
- `true` if the original caption explicitly mentioned "left" or "right" for this body part
- `false` if the laterality was not specified in the caption

## 3. original_laterality
- The original laterality from caption: "left", "right", or "unspecified"
- This preserves the original information for reference

## 4. text_span
- The exact text from the caption that describes this action
- Keep it minimal but complete enough to identify the action

## 5. action_description (for SAM grounding, ≤32 tokens)
- Describe the body part performing the action for visual grounding
- Format: "[person descriptor]'s [body part] [action context]"
- **Use normalized body part name (without left/right)**
- Example: "man's hand holding the cup", "woman's foot kicking the ball"

## 6. interaction_object (≤32 tokens)
- The object or person being interacted with
- Keep it SHORT and VISUALLY DISTINCTIVE (2-5 words)
- Set to `null` if no physical interaction (e.g., waving alone)

## 7. interaction_type
- One of: "holding", "touching", "pointing", "kicking", "carrying", "reaching", "waving", "grasping", "pushing", "pulling", "other"

## 8. person_id
- The person ID from the caption (e.g., "ID_A", "ID_B")

## 9. person_full_description
- **The COMPLETE description from <Role> section for this person**
- Copy the full text exactly as it appears in <Role>

## 10. person_brief_description (for SAM grounding, ≤10 tokens)
- **A SHORT but DISTINCTIVE description for visual grounding**
- Extract the MOST VISUALLY DISTINCTIVE features that differentiate this person from others
- Focus on: clothing color, unique accessories, or notable physical features
- Format: "[gender/age] in [most distinctive clothing/feature]"

# Output Format
**IMPORTANT: Output ONLY valid JSON. No explanation or markdown.**

```json
{{
    "uncertain_parts": [
        {{
            "person_id": "ID_A",
            "person_full_description": "A young Asian man wearing a blue polo shirt and khaki pants",
            "person_brief_description": "man in blue polo shirt",
            "body_part": "hand of man in blue polo shirt",
            "laterality_specified": true,
            "original_laterality": "left",
            "text_span": "holding a phone in his left hand",
            "action_description": "man's hand holding phone",
            "interaction_object": "phone",
            "interaction_type": "holding"
        }}
    ]
}}
```

# Field Descriptions Summary

| Field | Description | When to return empty/null |
|-------|-------------|---------------------------|
| `person_id` | Person ID from caption (e.g., "ID_A") | Never empty if extracting |
| `person_full_description` | Complete description from <Role> section | Never empty if extracting |
| `person_brief_description` | Short distinctive description (≤10 tokens) | Never empty if extracting |
| `body_part` | Format: "[body_part] of [person_brief_description]", normalized (no left/right) | Never empty if extracting |
| `laterality_specified` | Whether "left"/"right" was mentioned | Always boolean |
| `original_laterality` | "left", "right", or "unspecified" | Always one of these three |
| `text_span` | Exact text from caption | Never empty if extracting |
| `action_description` | Grounding description (≤32 tokens) | Never empty if extracting |
| `interaction_object` | Object being interacted with | `null` if no physical interaction |
| `interaction_type` | Type of interaction | Never empty if extracting |

# When to Return Empty List
Return `{{"uncertain_parts": []}}` when:
1. No body parts are involved in any actions
2. Only symmetric actions exist (clapping, folding arms)
3. Only "both hands/feet" actions on same object exist
4. Only non-lateral body parts are mentioned (head, face, back, chest)

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
            if 'uncertain_parts' in item and item['uncertain_parts']:
                continue
            
            caption = item.get('caption', '')
            if not caption:
                continue
            
            # 解析并保存人物ID映射到item中，供后续步骤使用
            id_descriptions = self._parse_role_section(caption)
            item['person_id_mapping'] = id_descriptions
            
            # 如果没有人物ID，直接标记为"no human in the picture"
            if not id_descriptions:
                item['uncertain_parts'] = "no_human_in_picture"
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
                    items[idx]['uncertain_parts'] = []
                    items[idx]['llm_parse_error'] = "failed_to_extract_json"
                    continue
                
                uncertain_parts = result.get("uncertain_parts", [])
                
                # 处理特殊情况
                if isinstance(uncertain_parts, str):
                    # LLM 返回了字符串（如 "all_body_parts_already_specified"）
                    items[idx]['uncertain_parts'] = uncertain_parts
                elif isinstance(uncertain_parts, list) and len(uncertain_parts) == 0:
                    # 有人物但没有不确定的身体部位
                    items[idx]['uncertain_parts'] = "all_body_parts_already_specified"
                else:
                    items[idx]['uncertain_parts'] = uncertain_parts
                    
            except Exception as e:
                print(f"Error parsing output for item {idx}: {e}")
                print(f"Raw output: {generated_text}...")
                items[idx]['uncertain_parts'] = []
                items[idx]['llm_parse_error'] = str(e)
        
        # Cleanup
        self.unload_model()
        return items

    def process_caption(self, caption):
        """Wrapper for single item processing."""
        item = {"caption": caption}
        self.process_batch([item])
        return {
            "uncertain_parts": item.get("uncertain_parts", []),
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
