import os
import json
import random
import numpy as np
import cv2
import torch
from tqdm import tqdm

filtered_dir = "/mydata/data/seunghoonjeong/co3dv2_filtered"
category_list = [category for category in os.listdir(filtered_dir) if os.path.isdir(os.path.join(filtered_dir, category))]
category_list.sort()

some_category_list = ["apple", "backpack", "bench", "book", "chair", "cup", "hydrant", "laptop", "teddybear"]

for category in some_category_list:
    category_dir = os.path.join(filtered_dir, category)
    seq_info_full_json_path = os.path.join(category_dir, "co3dv2_centric_seq_info_full.json")
    with open(seq_info_full_json_path, "r") as f:
        seq_info_full_dict = json.load(f)

    for split in ["train", "val", "test"]:
        print(f"Processing {category} {split}...")
        split_dataset = []
        split_seq_list_json_path = os.path.join(category_dir, f"{split}_seq_names.json")
        with open(split_seq_list_json_path, "r") as f:
            split_seq_list = json.load(f)

        for seq_name in split_seq_list:
            print(seq_name)
            seq_paired_data_json_path = os.path.join(category_dir, seq_name, "paired_data.json")
            with open(seq_paired_data_json_path, "r") as f:
                seq_paired_data = json.load(f)

            for data in seq_paired_data:
                image_name = data["warped_image_path"].split("/")[-1]
                ori_mask_image_name = image_name.split(".")[0] + "_mask.png"
                mask_image_name = "mask_" + image_name
                data["warped_image_path"] = data["warped_image_path"].replace("warped", "warped_aligned")
                data["warped_mask_path"] = data["warped_mask_path"].replace(ori_mask_image_name, mask_image_name)
                assert os.path.exists(data["warped_image_path"])
                assert os.path.exists(data["warped_mask_path"])
                split_dataset.append(data)

        split_dataset_json_path = os.path.join(category_dir, f"{split}_dataset.json")
        with open(split_dataset_json_path, "w") as f:
            json.dump(split_dataset, f, indent=4)
        