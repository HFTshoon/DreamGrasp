import os
import json

dataset_path = "/mydata/data/seunghoonjeong/co3dv2"
category_list = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]
category_list.sort()

seq_json = {}
for category in category_list:
    category_path = os.path.join(dataset_path, category)
    
    # seq if starts with digit
    seq_list = [seq for seq in os.listdir(category_path) if seq[0].isdigit()]
    
    # sort seq by number
    seq_list.sort(key=lambda x: int(x.split("_")[0]))
    
    seq_json[category] = seq_list

with open("/mydata/data/seunghoonjeong/DreamGrasp/co3dv2_seq.json", "w") as f:
    json.dump(seq_json, f, indent=4)
