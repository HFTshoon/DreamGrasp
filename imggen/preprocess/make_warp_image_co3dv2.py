import os
import json
import random

import torch
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
from einops import rearrange
from splatting import splatting_function

# import sys
# sys.path.append("recon")
# from mast3r.model import AsymmetricMASt3R

# from dust3r.inference import inference
# from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from util.util_preprocess import make_pts3d
from util.util_warp import compute_optical_flow
from util.util_point import estimate_normals_towards_camera

# model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
# device = 'cuda'
# batch_size = 1
# schedule = 'cosine'
# lr = 0.01
# niter = 300
# model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

def rotation_angle_rad(R_ref, R_current):
    R_rel = R_ref.T @ R_current

    val = (np.trace(R_rel) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)

    angle_rad = np.arccos(val)
    return angle_rad, 4 * val * (1-val)

filtered_dir = "/mydata/data/seunghoonjeong/co3dv2_filtered"
category_list = [category for category in os.listdir(filtered_dir) if os.path.isdir(os.path.join(filtered_dir, category))]
category_list.sort()

some_category_list = ["apple", "backpack", "bench", "book", "chair", "cup", "hydrant", "laptop", "teddybear"]
for category in some_category_list[1:]:
    category_dir = os.path.join(filtered_dir, category)
    refine_seq_name_json_path = os.path.join(category_dir, "co3dv2_centric_seq_name_refine.json")
    with open(refine_seq_name_json_path, "r") as f:
        refine_seq_names = json.load(f)

    seq_idx_json_path = os.path.join(category_dir, "co3dv2_centric_seq_info_mini.json")
    with open(seq_idx_json_path, "r") as f:
        seq_idx_dict = json.load(f)

    seq_cnt = len(refine_seq_names)
    random.seed(42)
    random.shuffle(refine_seq_names)
    train_seq_names = refine_seq_names[:int(seq_cnt * 0.8)]
    val_seq_names = refine_seq_names[int(seq_cnt * 0.8):int(seq_cnt * 0.9)]
    test_seq_names = refine_seq_names[int(seq_cnt * 0.9):]

    with open(os.path.join(category_dir, "train_seq_names.json"), "w") as f:
        json.dump(train_seq_names, f)
    with open(os.path.join(category_dir, "val_seq_names.json"), "w") as f:
        json.dump(val_seq_names, f)
    with open(os.path.join(category_dir, "test_seq_names.json"), "w") as f:
        json.dump(test_seq_names, f)

    for seq_name in tqdm(refine_seq_names):
        print(seq_name)
        seq_paired_data = []
        seq_path = os.path.join(category_dir, seq_name, "images")
        seq_list = os.listdir(seq_path)
        image_list = [f"frame{frame_idx:06d}.jpg" for frame_idx in seq_idx_dict[seq_name]]
        image_list.sort()

        warped_dir = os.path.join(category_dir, seq_name, "warped")
        if os.path.exists(warped_dir):
            os.system(f"rm -rf {warped_dir}")
        os.makedirs(warped_dir, exist_ok=True)

        poses = np.load(os.path.join(category_dir, seq_name, 'poses.npy'))
        focals = np.load(os.path.join(category_dir, seq_name, 'focals.npy'))
        pps = np.load(os.path.join(category_dir, seq_name, 'pps.npy'))
        depthmaps = np.load(os.path.join(category_dir, seq_name, 'depthmaps.npy'))
        pts3d = make_pts3d(depthmaps, poses, focals, pps)
        h,w = depthmaps[0].shape

        for query_idx in tqdm(range(len(seq_idx_dict[seq_name][::3]))):
            query_image_num = seq_idx_dict[seq_name][::3][query_idx]
            query_pose = poses[query_idx]
            query_info_path = os.path.join(seq_path, f'frame{query_image_num:06d}.npz')
            query_info = np.load(query_info_path)
            assert query_info["camera_pose"].all() == query_pose.all()
            query_focal = focals[query_idx]
            query_pp = pps[query_idx]
            query_R = query_pose[:3, :3]
            query_T = query_pose[:3, 3]
            query_K = torch.tensor([
                [query_focal[0], 0, query_pp[0]], 
                [0, query_focal[0], query_pp[1]], 
                [0, 0, 1]
            ]).float()

            angle_from_query = []
            score_from_query = []
            for ref_idx in range(len(seq_idx_dict[seq_name][::3])):
                if ref_idx == query_idx:
                    angle_from_query.append(0)
                    score_from_query.append((0,ref_idx))
                    continue
                
                ref_image_num = seq_idx_dict[seq_name][::3][ref_idx]
                ref_pose = poses[ref_idx]
                ref_R = ref_pose[:3, :3]
                angle, score = rotation_angle_rad(query_R, ref_R)
                angle_from_query.append(angle)
                score_from_query.append((score,ref_idx))
            score_from_query = np.array(score_from_query)

            positive_score = score_from_query[score_from_query[:,0] > 0]

            if len(positive_score) == 0:
                continue

            weights, values = zip(*positive_score)  # 각 튜플에서 weight와 value를 분리
            ref_idx_source = random.choices(values, weights=weights, k=3)
            ref_idx_source.sort()
            # ref_idx_candidates_list = [
            #     [int(ref_idx_source[0])],
            #     [int(ref_idx_source[1])],
            #     [int(ref_idx_source[2])],
            #     [int(ref_idx_source[0]), int(ref_idx_source[1])],
            #     [int(ref_idx_source[0]), int(ref_idx_source[2])],
            #     [int(ref_idx_source[1]), int(ref_idx_source[2])],
            #     [int(ref_idx_source[0]), int(ref_idx_source[1]), int(ref_idx_source[2])]
            # ]

            ref_idx_candidates_list = [
                [int(ref_idx_source[0])],
                [int(ref_idx_source[1])],
                [int(ref_idx_source[2])]
            ]

            for ref_idx_candidates in ref_idx_candidates_list:
                ref_idx_image_nums = [seq_idx_dict[seq_name][::3][ref_idx] for ref_idx in ref_idx_candidates]

                ref_images = []
                ref_flows = []
                ref_depths = []
                for ref_idx in ref_idx_candidates:
                    ref_image_num = seq_idx_dict[seq_name][::3][ref_idx]
                    ref_pose = poses[ref_idx]
                    ref_info_path = os.path.join(category_dir, seq_name, 'images', f'frame{ref_image_num:06d}.npz')
                    ref_info = np.load(ref_info_path)
                    assert ref_info["camera_pose"].all() == ref_pose.all()
                    ref_focal = focals[ref_idx]
                    ref_pp = pps[ref_idx]
                    ref_R = ref_pose[:3, :3]
                    ref_T = ref_pose[:3, 3]
                    ref_K = torch.tensor([
                        [ref_focal[0], 0, ref_pp[0]], 
                        [0, ref_focal[0], ref_pp[1]], 
                        [0, 0, 1]
                    ]).float()

                    # get flow and depth and normal
                    ref_pts = pts3d[ref_idx].reshape(-1, 3)
                    ref_normals = estimate_normals_towards_camera(
                        ref_pts.reshape(-1, 3), camera_pos=ref_T, radius=0.1, max_nn=30
                    )
                    ref_normals = torch.tensor(ref_normals).float()

                    ref_pts = torch.tensor(ref_pts).float()
                    flow, depth = compute_optical_flow(ref_pts, ref_R, ref_T, ref_K, query_R, query_T, query_K, ref_normals)
                    flow = flow.reshape(h, w, 2)

                    ref_image_path = os.path.join(category_dir, seq_name, 'images', f'frame{ref_image_num:06d}.jpg')
                    ref_image = imageio.imread(ref_image_path)
                    ref_image = torch.tensor(ref_image).float() / 127.5 - 1.0

                    image = rearrange(ref_image, 'h w c -> c h w').unsqueeze(0)
                    flow = rearrange(flow, 'h w c -> c h w').unsqueeze(0)
                    ref_images.append(image)
                    ref_flows.append(flow)
                    ref_depths.append(depth)

                    # # importance weight based on depth
                    # importance = 0.5 / depth
                    # importance -= importance.min()
                    # importance /= importance.max() + 1e-6
                    # importance = importance * 10 - 10
                    # importance = importance.reshape(h, w, 1)
                    # importance = rearrange(importance, 'h w c -> c h w').unsqueeze(0)
                    # warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
                    # mask = (warped == 0).all(dim=1, keepdim=True).to(image.dtype)
                    
                    # overlap = 1 - mask.sum().item() / (h*w)

                    # # visualize
                    # warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
                    # warped = ((warped + 1) * 127.5).astype(np.uint8)
                    # mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
                    # cv2.imwrite(os.path.join(warped_dir, f"{query_image_num}_{ref_image_num}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite(os.path.join(warped_dir, f"{query_image_num}_{ref_image_num}_mask.png"), (1-mask)*255)
                
                ref_cnt = len(ref_images)

                ref_images_stack = torch.cat(ref_images, dim=2)     # (1, 3, H * B, W)

                for i, flow in enumerate(ref_flows):
                    flow[0,1] -= i * h
                ref_flows_stack = torch.cat(ref_flows, dim=2)       # (1, 2, H * B, W)

                ref_depths_stack = torch.cat(ref_depths, dim=0)     # (H * W * B,)
                ref_importance = 0.5 / ref_depths_stack
                ref_importance -= ref_importance.min()
                ref_importance /= ref_importance.max() + 1e-6
                ref_importance = ref_importance * 10 - 10

                ref_importance_chunks = []
                for i in range(ref_cnt):
                    chunk = ref_importance[i*h*w:(i+1)*h*w]
                    chunk = chunk.reshape(h, w, 1)
                    chunk = rearrange(chunk, 'h w c -> c h w').unsqueeze(0)
                    ref_importance_chunks.append(chunk)
                ref_importance = torch.cat(ref_importance_chunks, dim=2)  # (1, 1, H * B, W)

                warped = splatting_function('softmax', ref_images_stack, ref_flows_stack, ref_importance, eps=1e-6)
                warped = warped[:,:,:h,:]  # (1, 3, H, W)
                mask = (warped == 0).all(dim=1, keepdim=True).to(image.dtype)
                mask = mask[:,:,:h,:]  # (1, 1, H, W)

                mask_ratio = mask.sum().item() / (h*w)
                if mask_ratio > 0.8:
                    print(f"Unvisible region ratio is too high: {mask_ratio} ({category}, {seq_name}, {query_image_num}, {ref_idx_image_nums})")
                    warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
                    warped = ((warped + 1) * 127.5).astype(np.uint8)
                    mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
                    candidate_str = "-".join([str(ref_idx) for ref_idx in ref_idx_image_nums])
                    cv2.imwrite(os.path.join(warped_dir, f"failed_{query_image_num}_{candidate_str}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(warped_dir, f"failed_mask_{query_image_num}_{candidate_str}.png"), (1-mask)*255)
                    continue    

                # visualize
                warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
                warped = ((warped + 1) * 127.5).astype(np.uint8)
                mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
                candidate_str = "-".join([str(ref_idx) for ref_idx in ref_idx_image_nums])
                cv2.imwrite(os.path.join(warped_dir, f"{query_image_num}_{candidate_str}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(warped_dir, f"mask_{query_image_num}_{candidate_str}.png"), (1-mask)*255)
                gt_image = os.path.join(category_dir, seq_name, 'images', f'frame{query_image_num:06d}.jpg')
                gt_image_save_path = os.path.join(warped_dir, f"{query_image_num}.jpg")
                os.system(f"cp {gt_image} {gt_image_save_path}")
                seq_paired_data.append({
                    "category": category,
                    "seq_name": seq_name,
                    "query_idx": query_idx,
                    "query_image_num": query_image_num,
                    "ref_idx_candidates": ref_idx_candidates,
                    "ref_image_nums": ref_idx_image_nums,
                    "ref_angles": [np.degrees(angle_from_query[ref_idx]) for ref_idx in ref_idx_candidates],
                    "mask_ratio": mask_ratio,
                    "warped_image_path": os.path.join(warped_dir, f"{query_image_num}_{candidate_str}.png"),
                    "warped_mask_path": os.path.join(warped_dir, f"{query_image_num}_{candidate_str}_mask.png"),
                })
        with open(os.path.join(category_dir, seq_name, "paired_data.json"), "w") as f:
            json.dump(seq_paired_data, f, indent=4)

            