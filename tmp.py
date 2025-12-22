import json
import os

sam_anno_path="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter_anno_sam.json"
sam_data_dir="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_sam_yolo_anno"

with open ("/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter_anno_sam copy.json","r",encoding="utf-8") as f:
    sam_anno_data=json.load(f)
    
    
sam_path_dict={}
for ranl in os.listdir(sam_data_dir):
    for image_path in os.listdir(os.path.join(sam_data_dir,ranl)):
        sam_path_dict[image_path.split('_')[-1]]=os.path.join(sam_data_dir,ranl,image_path)
        
    
    


for item in sam_anno_data:
    if item['image_path'].split('/')[-1] in sam_path_dict:
        item['image_path']=sam_path_dict[item['image_path'].split('/')[-1]]
        
with open(sam_anno_path,"w") as f:
    json.dump(sam_anno_data,f,indent=4,ensure_ascii=False)