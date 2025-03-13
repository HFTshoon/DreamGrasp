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

for category in tqdm(category_list):
    category_dir = os.path.join(filtered_dir, category)
    seq_info_full_json_path = os.path.join(category_dir, "co3dv2_centric_seq_info_full.json")
    with open(seq_info_full_json_path, "r") as f:
        seq_info_full_dict = json.load(f)

    seq_list = [seq_name for seq_name in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, seq_name))]
    for seq_name in seq_list:
        image_dir = os.path.join(category_dir, seq_name, "images")
        seq_info_full = seq_info_full_dict[seq_name]
        for frame_info in seq_info_full:
            frame_idx = frame_info["frame_idx"]
            file_path = frame_info["file_path"]
            camera_intrinsics = frame_info["camera_intrinsics"]
            camera_pose = frame_info["camera_pose"]
            npz_name = f"frame{frame_idx:06d}.npz"

            save_img_path = os.path.join("/mydata/data/seunghoonjeong/co3dv2_filtered", file_path)
            save_meta_path = save_img_path.replace('jpg', 'npz')
            np.savez(
                save_meta_path, 
                camera_intrinsics=camera_intrinsics,
                camera_pose=camera_pose
            )