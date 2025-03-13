import os
import json
import random
import numpy as np
import cv2
import torch
from tqdm import tqdm

import sys
sys.path.append("recon")
from mast3r.model import AsymmetricMASt3R

from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

filtered_dir = "/mydata/data/seunghoonjeong/co3dv2_filtered"
category_list = [category for category in os.listdir(filtered_dir) if os.path.isdir(os.path.join(filtered_dir, category))]
category_list.sort()

model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
device = 'cuda'
batch_size = 45
schedule = 'cosine'
lr = 0.01
niter = 300
model = AsymmetricMASt3R.from_pretrained(model_path).to(device)


some_category_list = ["apple", "backpack", "bench", "book", "chair", "cup", "hydrant", "laptop", "suitcase", "teddybear"]
for category in some_category_list:
    category_dir = os.path.join(filtered_dir, category)
    refine_seq_name_json_path = os.path.join(category_dir, "co3dv2_centric_seq_name_refine.json")
    with open(refine_seq_name_json_path, "r") as f:
        refine_seq_names = json.load(f)

    seq_idx_json_path = os.path.join(category_dir, "co3dv2_centric_seq_info_mini.json")
    with open(seq_idx_json_path, "r") as f:
        seq_idx_dict = json.load(f)

    for seq_name in tqdm(refine_seq_names):
        print("-------------------- Processing", seq_name, "--------------------")
        seq_path = os.path.join(category_dir, seq_name, "images")
        seq_list = os.listdir(seq_path)
        image_list = [f"frame{frame_idx:06d}.jpg" for frame_idx in seq_idx_dict[seq_name][::3]]
        image_list.sort()

        img_path_list = [os.path.join(seq_path, img_name) for img_name in image_list]
        images = load_images(img_path_list, size=512, square_ok=True)

        info_path_list = [os.path.join(seq_path, img_name.replace('.jpg', '.npz')) for img_name in image_list]
        info_list = [np.load(info_path) for info_path in info_path_list]
        input_poses = np.array([info['camera_pose'] for info in info_list])
        input_focals = np.array([info['camera_intrinsics'][0, 0] for info in info_list])

        input_poses = torch.tensor(input_poses).float().to(device)

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        try:
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            scene.min_conf_thr = 1.5 # for MASt3R
            
            scene.preset_pose(input_poses)
            scene.preset_focal(input_focals)        
            scene.preset_principal_point_zero()

            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            
            poses = scene.get_im_poses()            # (N, 4, 4)
            focals = scene.get_focals()             # (N, 1)
            pps = scene.get_principal_points()      # (N, 2)
            depthmaps = scene.get_depthmaps()       # N x (512, 512)
            pts3d = scene.get_pts3d()               # N X (H, W, 3)

            poses = poses.detach().cpu().numpy()
            focals = focals.detach().cpu().numpy()
        except:
            # delete this seq from selected_sequences_numbers_subset_dict
            print(f"Error in {seq_name}")
            continue

        N = len(image_list)
        depthmaps_numpy = np.zeros((N, 512, 512), dtype=np.float32)
        for i, dps in enumerate(depthmaps):
            dps = dps.detach().cpu().numpy()
            depthmaps_numpy[i] = dps

        pps = pps.detach().cpu().numpy()

        np.save(os.path.join(category_dir, seq_name, f"poses.npy"), poses)
        np.save(os.path.join(category_dir, seq_name, f"focals.npy"), focals)
        np.save(os.path.join(category_dir, seq_name, f"depthmaps.npy"), depthmaps_numpy)
        np.save(os.path.join(category_dir, seq_name, f"pps.npy"), pps)
