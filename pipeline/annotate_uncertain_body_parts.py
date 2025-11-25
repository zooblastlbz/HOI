import json
import os
import argparse
from openai import OpenAI
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

def get_uncertain_body_parts(client, model, caption):
    """
    调用 LLM 分析 caption 中不确定的身体部位及其交互物体。
    """
    prompt = f"""
    Task: Analyze the following image caption and identify human body parts where the spatial location (left/right) is NOT specified (uncertain).
    For each uncertain body part, identify the object it is interacting with, the exact text span, classify the interaction type, and identify which person it belongs to.

    Caption: "{caption}"

    Requirements:
    1. Only identify body parts belonging to humans.
    2. Only include body parts where 'left' or 'right' is missing in the description.
       - Example: "holding a cup with hand" -> "hand" is uncertain.
       - Example: "holding a cup with left hand" -> NOT uncertain.
       - Example: "holding a bag in the other hand" (if context implies left/right is unknown) -> "hand" is uncertain.
    3. Identify the person to whom the body part belongs. This is crucial for multi-person scenarios.
       - Extract the specific phrase describing the person (e.g., "A man", "The woman in the red shirt", "The boy").
    4. Classify the interaction into categories to help decide the visual grounding strategy. If the situation is ambiguous or fits both descriptions, include BOTH categories:
       - "object_interaction": The body part is actively manipulating, touching, or pointing at a specific object. (e.g., holding a cup, kicking a ball, touching a table, pointing at a sign). This implies we can use object grounding (detecting the object) to help identify the body part (left vs right).
       - "spatial_proximity": The body part is described by its spatial proximity to an object or its general posture/position, without active manipulation. (e.g., hand near the wall, arm raised, leg bent, standing next to a car). This implies we should rely more on pose/skeleton estimation to determine the side.
    5. Output a JSON object with a key "uncertain_parts" containing a list of objects.
    6. Each object in the list should have:
       - "body_part": The body part mentioned (e.g., "hand", "leg", "arm", "torso").
       - "person_description": The text description of the person this body part belongs to (e.g., "A man", "The girl").
       - "interaction_object": The object interacting with the body part (or null if none).
       - "text_span": The exact phrase or segment in the caption that refers to this action/body part (e.g., "holding a cup with his hand").
       - "interaction_category": A LIST containing one or both of ["object_interaction", "spatial_proximity"].
    
    Example Output:
    {{
        "uncertain_parts": [
            {{
                "body_part": "hand", 
                "person_description": "A man",
                "interaction_object": "cup",
                "text_span": "holding a cup with his hand",
                "interaction_category": ["object_interaction"]
            }},
            {{
                "body_part": "foot", 
                "person_description": "The woman in red",
                "interaction_object": "soccer ball",
                "text_span": "kicking a soccer ball",
                "interaction_category": ["object_interaction"]
            }}
        ]
    }}

    Return ONLY the JSON string.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error processing caption: {caption[:30]}... Error: {e}")
        return {"uncertain_parts": []}

def main():
    parser = argparse.ArgumentParser(description="Annotate uncertain body parts in image captions using LLM.")
    parser.add_argument("--api-key", type=str, help="OpenAI API Key. Defaults to OPENAI_API_KEY env var.")
    parser.add_argument("--base-url", type=str, help="OpenAI Base URL. Defaults to OPENAI_BASE_URL env var.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use (default: gpt-4o).")
    parser.add_argument("--input", type=str, default="data.json", help="Input JSON file path (default: data.json).")
    parser.add_argument("--output", type=str, default="data_annotated.json", help="Output JSON file path (default: data_annotated.json).")
    
    args = parser.parse_args()

    # 配置 OpenAI Client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    
    if not api_key:
        print("Warning: No API Key provided. Ensure you have set OPENAI_API_KEY or passed --api-key.")
        # 某些本地模型可能不需要 key，但通常 OpenAI SDK 需要非空值
        api_key = "dummy" 

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 设定输入输出路径
    # 如果是相对路径，则相对于当前工作目录
    input_file = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        # 为了演示代码逻辑，这里创建一个假的 data.json
        print("Creating a dummy data.json for demonstration purposes...")
        dummy_data = [
            {"id": 1, "caption": "A man holding a cup with his hand."},
            {"id": 2, "caption": "A woman kicking a soccer ball."},
            {"id": 3, "caption": "He is touching the table with his left hand and holding a pen in the other hand."}
        ]
        with open(input_file, 'w') as f:
            json.dump(dummy_data, f, indent=2)
    
    print(f"Reading from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # 处理数据
    iterator = tqdm(data, desc="Processing captions")
    
    for item in iterator:
        # 假设 caption 字段名为 'caption'，请根据实际数据修改
        caption = item.get('caption', '')
        
        if caption:
            annotation = get_uncertain_body_parts(client, args.model, caption)
            # 将标注结果合并到原数据中
            item['uncertain_body_part_annotation'] = annotation.get('uncertain_parts', [])
        else:
            item['uncertain_body_part_annotation'] = []
            
        results.append(item)
        
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()
