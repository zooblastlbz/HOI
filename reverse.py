
import json




rewrite_anno_path="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter_anno_sam_rewrite.json"
origin_path_data="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter.json"

origin_path_dicrt={}

duplicate_dict={}
with open(origin_path_data,"r") as f:
    origin_data=json.load(f)
for item in origin_data:
    if item["image_path"].split("/")[-1] not in  origin_path_dicrt:
        origin_path_dicrt[item['image_path'].split("/")[-1]]=item["image_path"]
    else:
        if item["image_path"].split("/")[-1] not in duplicate_dict:
            duplicate_dict[item["image_path"].split("/")[-1]]=2
        else:
            duplicate_dict[item["image_path"].split("/")[-1]]+=1

total_deplicate=0
for key,value in duplicate_dict.items():
    total_deplicate+=value

print(total_deplicate)

with open(rewrite_anno_path,"r") as f:
    rewrite_data=json.load(f)

modified_rewrite_data=[]
for item in rewrite_data['results']:
    path=item['image_path'].split("_")[-1]
    if path not in duplicate_dict and path in origin_path_dicrt:
        item['image_path']=origin_path_dicrt[path]
        modified_rewrite_data.append(item)
        
print(len(modified_rewrite_data))

with open("/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/to_rewrite_20251223_yolo_filter_anno_sam_rewrite_modified.json","w") as f:
    json.dump(modified_rewrite_data,f,indent=2,ensure_ascii=False)