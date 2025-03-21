import os
import sys
import argparse
import json
import random

import torch
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
from einops import rearrange
from splatting import splatting_function

from imggen.preprocess.preprocess_co3d import CATEGORIES, CATEGORIES_IDX, SINGLE_SEQUENCE_CATEGORIES
from util.util_preprocess import make_pts3d
from util.util_warp import compute_optical_flow

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None) # "apple") # default=None)
    parser.add_argument('--single_sequence_subset', default=False, action='store_true',
                        help="prepare the single_sequence_subset instead.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess_dir", type=str, default='/mydata/data/seunghoonjeong/co3d_apple_preprocess')
    return parser

def make_paired_data(category_name, category_dir, category_selected_sequences_info, split, seed):
    random.seed(seed)
    category_paired_data = {
        "data": []
    }

    category_seq_name = category_selected_sequences_info.keys()

    for seq_name in tqdm(category_seq_name):
        category_seq_index = category_selected_sequences_info[seq_name]
        seq_dir = os.path.join(category_dir, seq_name)
        warped_dir = os.path.join(seq_dir, 'warped')
        os.makedirs(warped_dir, exist_ok=True)

        image_cnt = len(category_seq_index)
        poses = np.load(os.path.join(category_dir, seq_name, f'poses_{split}.npy'))
        focals = np.load(os.path.join(category_dir, seq_name, f'focals_{split}.npy'))
        pps = np.load(os.path.join(category_dir, seq_name, f'pps_{split}.npy'))
        depthmaps = np.load(os.path.join(category_dir, seq_name, f'depthmaps_{split}.npy'))
        depthmaps = depthmaps[:image_cnt, :, :]
        pts3d = make_pts3d(depthmaps, poses, focals, pps)
        h,w = depthmaps[0].shape

        for query_idx in range(len(category_seq_index)):
            query_image_num = category_seq_index[query_idx]
            query_pose = poses[query_idx]
            query_info_path = os.path.join(seq_dir, 'images', f'frame{query_image_num:06d}.npz')
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

            iteration = 10
            ref_history = []
            for _ in range(iteration):
                while True:
                    # choose how many reference images to use (1~3)
                    ref_cnt = random.randint(1, 3)

                    # get random 3 reference_idx without query_idx
                    ref_idx_candidates = list(range(len(category_seq_index)))
                    ref_idx_candidates.remove(query_idx)
                    ref_idx_candidates = random.sample(ref_idx_candidates, ref_cnt)
                    ref_idx_candidates.sort()
                    if ref_idx_candidates not in ref_history:
                        ref_history.append(ref_idx_candidates)
                        break

                ref_idx_image_nums = [category_seq_index[ref_idx] for ref_idx in ref_idx_candidates]

                ref_images = []
                ref_flows = []
                ref_depths = []
                for ref_idx in ref_idx_candidates:
                    ref_image_num = category_seq_index[ref_idx]
                    ref_pose = poses[ref_idx]
                    ref_info_path = os.path.join(seq_dir, 'images', f'frame{ref_image_num:06d}.npz')
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

                    # get flow and depth
                    ref_pts = pts3d[ref_idx].reshape(-1, 3)
                    ref_pts = torch.tensor(ref_pts).float()
                    flow, depth = compute_optical_flow(ref_pts, ref_R, ref_T, ref_K, query_R, query_T, query_K)
                    flow = flow.reshape(h, w, 2)

                    ref_image_path = os.path.join(seq_dir, 'images', f'frame{ref_image_num:06d}.jpg')
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
                    # cv2.imwrite(os.path.join(seq_dir, 'warped', f"{query_image_num}_{ref_image_num}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite(os.path.join(seq_dir, 'warped', f"{query_image_num}_{ref_image_num}_mask.png"), (1-mask)*255)
                
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
                    print(f"Unvisible region ratio is too high: {mask_ratio} ({category_name}, {seq_name}, {query_image_num}, {ref_idx_image_nums})")
                    warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
                    warped = ((warped + 1) * 127.5).astype(np.uint8)
                    mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
                    candidate_str = "-".join([str(ref_idx) for ref_idx in ref_idx_image_nums])
                    cv2.imwrite(os.path.join(warped_dir, f"failed_{query_image_num}_{candidate_str}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(warped_dir, f"failed_{query_image_num}_{candidate_str}_mask.png"), (1-mask)*255)
                    continue    

                # visualize
                warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
                warped = ((warped + 1) * 127.5).astype(np.uint8)
                mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
                candidate_str = "-".join([str(ref_idx) for ref_idx in ref_idx_image_nums])
                cv2.imwrite(os.path.join(warped_dir, f"{query_image_num}_{candidate_str}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(warped_dir, f"{query_image_num}_{candidate_str}_mask.png"), (1-mask)*255)
                category_paired_data["data"].append({
                    "category": category_name,
                    "seq_name": seq_name,
                    "query_idx": query_idx,
                    "query_image_num": query_image_num,
                    "ref_idx_candidates": ref_idx_candidates,
                    "ref_image_nums": ref_idx_image_nums,
                    "warped_image_path": os.path.join(warped_dir, f"{query_image_num}_{candidate_str}.png"),
                    "warped_mask_path": os.path.join(warped_dir, f"{query_image_num}_{candidate_str}_mask.png"),
                })

    return category_paired_data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.category is None:
        if args.single_sequence_subset:
            categories = SINGLE_SEQUENCE_CATEGORIES
        else:
            categories = CATEGORIES
    else:
        categories = [args.category]

    for split in ['train', 'test']:
    # for split in ['train']:
        paired_data_path = os.path.join(args.preprocess_dir, f'paired_data_{split}.json')
        if os.path.isfile(paired_data_path):
            continue

        paired_data = {}
        selected_sequences_path = os.path.join(args.preprocess_dir, f'selected_seqs_{split}.json')
        with open(selected_sequences_path, 'r') as f:
            selected_sequences_info = json.load(f)

        for category in categories:
            category_dir = os.path.join(args.preprocess_dir, category)
            category_paired_data_path = os.path.join(category_dir, f'paired_data_{split}.json')
            if os.path.isfile(category_paired_data_path):
                with open(category_paired_data_path, 'r') as f:
                    category_paired_data = json.load(f)
            else:
                print(f"Processing {split} - category = {category}")
                category_selected_sequences_info = selected_sequences_info[category]
                category_paired_data = make_paired_data(
                    category_name=category,
                    category_dir=category_dir,
                    category_selected_sequences_info=category_selected_sequences_info,
                    split=split,
                    seed=args.seed + CATEGORIES_IDX[category],
                )
                with open(category_paired_data_path, 'w') as f:
                    json.dump(category_paired_data, f, indent=4)
                print(f"Saved {split} {category} paired data at {category_paired_data_path}")

            paired_data[category] = category_paired_data["data"]
        
        with open(paired_data_path, 'w') as f:
            json.dump(paired_data, f, indent=4)
        print(f"Saved {split} paired data at {paired_data_path}")