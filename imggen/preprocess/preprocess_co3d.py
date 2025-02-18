#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the CO3D dataset.
# Usage:
# python3 datasets_preprocess/preprocess_co3d.py --co3d_dir /path/to/co3d
# --------------------------------------------------------

import argparse
import random
import gzip
import json
import os
import os.path as osp

import torch
import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import util.util_preprocess as cropping  # noqa

from recon.mast3r.model import AsymmetricMASt3R

from recon.dust3r.inference import inference
from recon.dust3r.utils.image import load_images
from recon.dust3r.image_pairs import make_pairs
from recon.dust3r.cloud_opt import global_aligner, GlobalAlignerMode


CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave",
    "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign",
    "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv",
    "umbrella", "vase", "wineglass",
]
CATEGORIES_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}  # for seeding

SINGLE_SEQUENCE_CATEGORIES = sorted(set(CATEGORIES) - set(["microwave", "stopsign", "tv"]))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument('--single_sequence_subset', default=False, action='store_true',
                        help="prepare the single_sequence_subset instead.")
    parser.add_argument("--output_dir", type=str, default='/mydata/data/seunghoonjeong/co3d_sample_preprocess') # default='/mydata/data/seunghoonjeong/co3d_preprocess')
    parser.add_argument("--co3d_dir", type=str, default='/mydata/data/hyunsoo/co3d_sample') # default='/mydata/data/hyunsoo/co3d_new')
    parser.add_argument("--num_sequences_per_object", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_quality", type=float, default=0.5, help="Minimum viewpoint quality score.")

    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser


def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    # input:
    # R = [[-0.99935228, -0.03595155, -0.00158081],
    #    [ 0.03594376, -0.99934274,  0.00471025],
    #    [-0.00174911,  0.00465037,  0.99998766]]
    # T = [ 2.20334458,  1.63462281, 15.91346645]
    # focal = [3.45257282, 3.45257282]
    # p0 = [-0., -0.]
    # image_size = [ 719, 1282]

    # output:
    # R[0] = [[ 0.9994, -0.0359,  0.0017],
    #     [ 0.0360,  0.9993, -0.0047],
    #     [-0.0016,  0.0047,  1.0000]]
    # tvec[0] = [-2.2033, -1.6346, 15.9135]
    # camera_matrix[0] = [[1.2412e+03, 0.0000e+00, 6.4100e+02],
    #     [0.0000e+00, 1.2412e+03, 3.5950e+02],
    #     [0.0000e+00, 0.0000e+00, 1.0000e+00]]

    R = torch.from_numpy(R)[None, :, :]
    T = torch.from_numpy(T)[None, :]
    focal = torch.from_numpy(focal)[None, :]
    p0 = torch.from_numpy(p0)[None, :]
    image_size = torch.from_numpy(image_size)[None, :]

    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # R_pytorch3d = [[[ 0.9994,  0.0360, -0.0016],
    #     [-0.0359,  0.9993,  0.0047],
    #     [ 0.0017, -0.0047,  1.0000]]]
    # T_pytorch3d = [[-2.2033, -1.6346, 15.9135]]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))
    # image_size_wh = [[1282, 719]]

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0
    # c0 = [[641.0000, 359.5000]]

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale
    # principal_point = [[641.0000, 359.5000]]
    # focal_length = [[1241.1999, 1241.1999]]

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    # camera_matrix = [[[1.2412e+03, 0.0000e+00, 6.4100e+02],
    #     [0.0000e+00, 1.2412e+03, 3.5950e+02],
    #     [0.0000e+00, 0.0000e+00, 1.0000e+00]]]

    # R = [[ 0.9994, -0.0359,  0.0017],
    #     [ 0.0360,  0.9993, -0.0047],
    #     [-0.0016,  0.0047,  1.0000]]
    # tvec = [[-2.2033, -1.6346, 15.9135]]
    return R[0], tvec[0], camera_matrix[0]


def get_set_list(category_dir, split, is_single_sequence_subset=False):
    sequences_all = []
    listfile = osp.join(category_dir, "set_lists.json")
    with open(listfile) as f:
        set_lists_data = json.load(f)
        sequences_all = set_lists_data[f"{split}_known"]
        sequences_all.extend(set_lists_data[f"{split}_unseen"])

    return sequences_all


def prepare_sequences(model, category, co3d_dir, output_dir, img_size, split, min_quality, max_num_sequences_per_object,
                      seed, is_single_sequence_subset=False):
    random.seed(seed)
    category_dir = osp.join(co3d_dir, category)
    category_output_dir = osp.join(output_dir, category)
    sequences_all = get_set_list(category_dir, split, is_single_sequence_subset)
    sequences_numbers = sorted(set(seq_name for seq_name, _, _ in sequences_all))

    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data

    good_quality_sequences = set()
    for seq_data in sequence_data:
        if seq_data["viewpoint_quality_score"] > min_quality:
            good_quality_sequences.add(seq_data["sequence_name"])

    sequences_numbers = [seq_name for seq_name in sequences_numbers if seq_name in good_quality_sequences]
    if len(sequences_numbers) < max_num_sequences_per_object:
        selected_sequences_numbers = sequences_numbers
    else:
        selected_sequences_numbers = random.sample(sequences_numbers, max_num_sequences_per_object)

    selected_sequences_numbers_dict = {seq_name: [] for seq_name in selected_sequences_numbers}
    selected_sequences_numbers_subset_dict = {seq_name: [] for seq_name in selected_sequences_numbers}
    sequences_all = [(seq_name, frame_number, filepath)
                     for seq_name, frame_number, filepath in sequences_all
                     if seq_name in selected_sequences_numbers_dict]
    
    for seq_name, frame_number, filepath in sequences_all:
        frame_idx = int(filepath.split('/')[-1][5:-4])
        selected_sequences_numbers_dict[seq_name].append(frame_idx)

    for selected_seq_name in selected_sequences_numbers:
        numbers = selected_sequences_numbers_dict[selected_seq_name]
        selected_sequences_numbers_subset_dict[selected_seq_name] = random.sample(numbers, 10)

    for seq_name, frame_number, filepath in tqdm(sequences_all):
        frame_idx = int(filepath.split('/')[-1][5:-4])
        
        if frame_idx not in selected_sequences_numbers_subset_dict[seq_name]:
            continue

        # selected_sequences_numbers_dict[seq_name].append(frame_idx)
        frame_data = frame_data_processed[seq_name][frame_number]
        focal_length = frame_data["viewpoint"]["focal_length"]
        principal_point = frame_data["viewpoint"]["principal_point"]
        image_size = frame_data["image"]["size"]

        K = convert_ndc_to_pinhole(focal_length, principal_point, image_size)
        R, tvec, camera_intrinsics = opencv_from_cameras_projection(np.array(frame_data["viewpoint"]["R"]),
                                                                    np.array(frame_data["viewpoint"]["T"]),
                                                                    np.array(focal_length),
                                                                    np.array(principal_point),
                                                                    np.array(image_size))

        frame_data = frame_data_processed[seq_name][frame_number]
        image_path = os.path.join(co3d_dir, filepath)

        input_rgb_image = PIL.Image.open(image_path).convert('RGB')

        W, H = input_rgb_image.size

        camera_intrinsics = camera_intrinsics.numpy()
        cx, cy = camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        min_margin = min(min_margin_x, min_margin_y)

        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin, cy - min_margin
        r, b = cx + min_margin, cy + min_margin
        crop_bbox = (l, t, r, b)
        input_rgb_image, input_camera_intrinsics = cropping.crop_image(
            input_rgb_image, camera_intrinsics, crop_bbox)

        output_resolution = np.array([img_size, img_size])

        input_rgb_image, input_camera_intrinsics = cropping.rescale_image(
            input_rgb_image, input_camera_intrinsics, output_resolution)
        
        # generate and adjust camera pose
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = tvec
        camera_pose = np.linalg.inv(camera_pose)

        # save crop images and depth, metadata
        save_img_path = os.path.join(output_dir, filepath)
        os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)

        input_rgb_image.save(save_img_path)

        save_meta_path = save_img_path.replace('jpg', 'npz')
        np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                 camera_pose=camera_pose)
        breakpoint()

    for seq_name in selected_sequences_numbers:
        seq_path = os.path.join(category_output_dir, seq_name, "images")
        seq_list = os.listdir(seq_path)
        image_list = [f for f in seq_list if f.endswith('.jpg')]
        image_list.sort()

        assert len(image_list) == 10

        img_path_list = [os.path.join(seq_path, img_name) for img_name in image_list]
        images = load_images(img_path_list, size=512, square_ok=True)

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.min_conf_thr = 1.5
        scene.preset_principal_point_zero()
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        
        poses = scene.get_im_poses()            # (10, 4, 4)
        focals = scene.get_focals()             # (10, 1)
        # pts3d = scene.get_pts3d()               # 10 x (512, 512, 3)
        pps = scene.get_principal_points()      # (10, 2)
        depthmaps = scene.get_depthmaps()       # 10 x (512, 512)

        poses = poses.detach().cpu().numpy()
        focals = focals.detach().cpu().numpy()

        depthmaps_numpy = np.zeros((10, 512, 512), dtype=np.float32)
        for i, dps in enumerate(depthmaps):
            dps = dps.detach().cpu().numpy()
            depthmaps_numpy[i] = dps

        pps = pps.detach().cpu().numpy()

        np.save(os.path.join(category_output_dir, seq_name, f"poses.npy"), poses)
        np.save(os.path.join(category_output_dir, seq_name, f"focals.npy"), focals)
        # np.save(os.path.join(category_output_dir, seq_name, f"pts3d.npy"), pts3d_numpy)
        np.save(os.path.join(category_output_dir, seq_name, f"depthmaps.npy"), depthmaps_numpy)
        np.save(os.path.join(category_output_dir, seq_name, f"pps.npy"), pps)
        # np.savez(os.path.join(category_output_dir, seq_name, f'recon.npz'), poses=poses, focals=focals, pts3d=pts3d_numpy, pps=pps)

    return selected_sequences_numbers_subset_dict


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.co3d_dir != args.output_dir
    if args.category is None:
        if args.single_sequence_subset:
            categories = SINGLE_SEQUENCE_CATEGORIES
        else:
            categories = CATEGORIES
    else:
        categories = [args.category]
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    for split in ['train', 'test']:
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path):
            continue

        all_selected_sequences = {}
        for category in categories:
            category_output_dir = osp.join(args.output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            category_selected_sequences_path = os.path.join(category_output_dir, f'selected_seqs_{split}.json')
            if os.path.isfile(category_selected_sequences_path):
                with open(category_selected_sequences_path, 'r') as fid:
                    category_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - category = {category}")
                category_selected_sequences = prepare_sequences(
                    model=model,
                    category=category,
                    co3d_dir=args.co3d_dir,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    min_quality=args.min_quality,
                    max_num_sequences_per_object=args.num_sequences_per_object,
                    seed=args.seed + CATEGORIES_IDX[category],
                    is_single_sequence_subset=args.single_sequence_subset
                )
                with open(category_selected_sequences_path, 'w') as file:
                    json.dump(category_selected_sequences, file)

            all_selected_sequences[category] = category_selected_sequences
        with open(selected_sequences_path, 'w') as file:
            json.dump(all_selected_sequences, file)
