import json

from aem import con

path="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_bbox_sharegpt_eval.json"



with open(path, 'r') as f:
    data = json.load(f)
    
origin_data="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_dataset.json"
with open(origin_data, 'r') as f:
    origin = json.load(f)
    
origin_caption_dict = {}
for item in origin:
    origin_caption=item['old_caption']
    origin_caption=origin_caption.split("<question_answer_split_token>")[-1]
    origin_caption=origin_caption.split("<|im_end|>")[0].strip()
    origin_caption_dict[item['caption']] = origin_caption

for item in data:
    content=item['messages'][1]['content']
    content_list=content.split("\n\n")
    new_content=content_list[0] + "\n\n"+origin_caption_dict[content_list[1]] + "\n\n" + content_list[2]
    item['messages'][1]['content'] = new_content
    
with open("/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_bbox_sharegpt_eval_origin_caption.json", 'w') as f:
    json.dump(data, f, indent=4)
    
