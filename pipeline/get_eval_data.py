import json
file_1="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_dataset_success.json"
file_2="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_dataset.json"

success_data=json.load(open(file_1,'r'))
all_data=json.load(open(file_2,'r'))

eval_data={'results':[]}

success_data_set=set()
for item in success_data['results']:
    original_caption=item['original_caption']
    if item.get('success', True):
        success_data_set.add(original_caption)
for item in all_data['results']:
    original_caption=item['original_caption']
    if item.get('success', True):
        if original_caption not in success_data_set:
            eval_data['results'].append(item)
        
with open('/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/kling_imgcap_100w_yolo_filtered_recaped_anno_rewrite_dataset_eval_no_cycle.json','w') as f:
    json.dump(eval_data,f,indent=2)

