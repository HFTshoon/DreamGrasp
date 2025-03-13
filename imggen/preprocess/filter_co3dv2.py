import os
import json
import gzip
import random

import torch
import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm

import util.util_preprocess as cropping  # noqa

def get_set_list(category_dir):
    listfiles = os.listdir(os.path.join(category_dir, "set_lists"))
    subset_list_files = [f for f in listfiles if f"fewview_train" in f]

    sequences_all = []
    for subset_list_file in subset_list_files:
        with open(os.path.join(category_dir, "set_lists", subset_list_file)) as f:
            subset_lists_data = json.load(f)
            sequences_all.extend(subset_lists_data["train"])
            sequences_all.extend(subset_lists_data["val"])
            sequences_all.extend(subset_lists_data["test"])

    return sequences_all

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

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

def farthest_point_sampling(points, M):
    np.random.seed(42)
    N = len(points)
    
    if M >= N:
        return list(range(N))

    first_index = np.random.randint(0, N)
    selected_indices = [first_index]
    min_dists = np.linalg.norm(points - points[first_index], axis=1)

    for _ in range(M - 1):
        next_index = np.argmax(min_dists)
        selected_indices.append(int(next_index))
        new_dists = np.linalg.norm(points - points[next_index], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected_indices

def rotation_angle_deg(R_ref, R_current):
    R_rel = R_ref.T @ R_current

    val = (np.trace(R_rel) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)

    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def check_camera_rotation_180(R_list):
    if len(R_list) == 0:
        return -1

    R_ref = R_list[0]
    max_deg = 0
    for i, R_current in enumerate(R_list):
        deg = rotation_angle_deg(R_ref, R_current)
        if deg > max_deg:
            max_deg = deg
        if deg >= 150:
            return i  # i번째 프레임에서 180도 이상 회전
    print(max_deg)
    return -1  # 한 번도 180도 이상 돌아간 적 없음

dataset_dir = "/mydata/data/seunghoonjeong/co3dv2"
filtered_dir = "/mydata/data/seunghoonjeong/co3dv2_filtered"
category_list = [category for category in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, category))]
category_list.sort()

# centric_seq_names_all = {}
# centric_seq_info_mini_all = {}
# centric_seq_info_full_all = {}
for category in category_list:
    if os.path.exists(os.path.join(filtered_dir, category, "co3dv2_centric_seq_info_full.json")):
        print("skip ", category)
        continue

    category_dir = os.path.join(dataset_dir, category)
    
    # seq if starts with digit
    seq_list = [seq for seq in os.listdir(category_dir) if seq[0].isdigit()]
    seq_list.sort(key=lambda x: int(x.split("_")[0]))
    
    with gzip.open(os.path.join(category_dir, "frame_annotations.jgz"), "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(os.path.join(category_dir, "sequence_annotations.jgz"), "r") as fin:
        sequence_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data

    good_quality_seqs = set()
    for seq_data in sequence_data:
        if seq_data["viewpoint_quality_score"] > 0.5:
            good_quality_seqs.add(seq_data["sequence_name"])

    seq_names = [seq_name for seq_name in seq_list if seq_name in good_quality_seqs]
    random.seed(42)
    seq_names = random.sample(seq_names, min(50, len(seq_names)))
    # seq_names = random.sample(seq_names, min(1, len(seq_names)))
    seq_names.sort()
    print(len(seq_names))
    
    seq_info_mini = {seq_name: [] for seq_name in seq_names}
    seq_info_full = {seq_name: [] for seq_name in seq_list}
    sequences_all = get_set_list(category_dir)
    sequences_all = [(seq_name, frame_number, filepath)
                     for seq_name, frame_number, filepath in sequences_all
                     if seq_name in seq_names]
    
    for seq_name, frame_number, filepath in tqdm(sequences_all):
        if seq_name not in seq_info_mini:
            continue
        
        frame_idx = int(filepath.split('/')[-1][5:-4])
        
        if frame_idx in seq_info_mini[seq_name]:
            continue
        
        frame_data = frame_data_processed[seq_name][frame_number]
        focal_length = frame_data["viewpoint"]["focal_length"]
        principal_point = frame_data["viewpoint"]["principal_point"]
        image_size = frame_data["image"]["size"]

        K = convert_ndc_to_pinhole(focal_length, principal_point, image_size)
        R, tvec, ori_camera_intrinsics = opencv_from_cameras_projection(np.array(frame_data["viewpoint"]["R"]),
                                                                    np.array(frame_data["viewpoint"]["T"]),
                                                                    np.array(focal_length),
                                                                    np.array(principal_point),
                                                                    np.array(image_size))

        image_path = os.path.join(dataset_dir, filepath)
        input_rgb_image = PIL.Image.open(image_path).convert('RGB')

        W, H = input_rgb_image.size

        ori_camera_intrinsics = ori_camera_intrinsics.numpy()
        cx, cy = ori_camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        min_margin = min(min_margin_x, min_margin_y)

        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin, cy - min_margin
        r, b = cx + min_margin, cy + min_margin
        crop_bbox = (l, t, r, b)
        input_rgb_image, input_camera_intrinsics = cropping.crop_image(
            input_rgb_image, ori_camera_intrinsics, crop_bbox)

        output_resolution = np.array([512, 512])

        input_rgb_image, input_camera_intrinsics = cropping.rescale_image(
            input_rgb_image, input_camera_intrinsics, output_resolution)
        
        # generate and adjust camera pose
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = tvec
        camera_pose = np.linalg.inv(camera_pose)

        seq_info_mini[seq_name].append(frame_idx)
        seq_info_full[seq_name].append({
            "frame_idx": frame_idx,
            "file_path": filepath,
            "image_path": image_path,
            "ori_camera_intrinsics": ori_camera_intrinsics.tolist(),
            "camera_intrinsics": input_camera_intrinsics.tolist(),
            "camera_pose": camera_pose.tolist()
        })

    centric_seq_names = []
    centric_seq_info_mini = {}
    centric_seq_info_full = {}
    for seq_name in seq_names:
        seq_Ts = []
        for frame_info in seq_info_full[seq_name]:
            camera_pose = np.array(frame_info["camera_pose"])
            seq_Ts.append(camera_pose[:3, 3])
        seq_Ts = np.array(seq_Ts)
        farthest_idx = farthest_point_sampling(seq_Ts, 30)
        idx_list = seq_info_mini[seq_name]
        info_list = seq_info_full[seq_name]
        seq_info_mini[seq_name] = [idx_list[idx] for idx in farthest_idx]
        seq_info_mini[seq_name] = sorted(seq_info_mini[seq_name])
        seq_info_full[seq_name] = [info_list[idx] for idx in farthest_idx]
        seq_info_full[seq_name] = sorted(seq_info_full[seq_name], key=lambda x: x["frame_idx"])
        
        seq_Rs = []
        for frame_info in seq_info_full[seq_name]:
            camera_pose = np.array(frame_info["camera_pose"])
            seq_Rs.append(camera_pose[:3, :3])
        is_360_seq = check_camera_rotation_180(seq_Rs)
        
        if is_360_seq != -1:
            print(seq_name)
            centric_seq_names.append(seq_name)
            centric_seq_info_mini[seq_name] = seq_info_mini[seq_name]
            centric_seq_info_full[seq_name] = seq_info_full[seq_name]
            for frame_info in seq_info_full[seq_name]:
                image_path = frame_info["image_path"]
                ori_camera_intrinsics = np.array(frame_info["ori_camera_intrinsics"])
                camera_intrinsics = np.array(frame_info["camera_intrinsics"])
                filepath = frame_info["file_path"]
                camera_pose = np.array(frame_info["camera_pose"])
                
                input_rgb_image = PIL.Image.open(image_path).convert('RGB')
                W, H = input_rgb_image.size

                cx, cy = ori_camera_intrinsics[:2, 2].round().astype(int)
                min_margin_x = min(cx, W - cx)
                min_margin_y = min(cy, H - cy)
                min_margin = min(min_margin_x, min_margin_y)

                # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
                l, t = cx - min_margin, cy - min_margin
                r, b = cx + min_margin, cy + min_margin
                crop_bbox = (l, t, r, b)
                input_rgb_image, input_camera_intrinsics = cropping.crop_image(
                    input_rgb_image, ori_camera_intrinsics, crop_bbox)

                output_resolution = np.array([512, 512])

                input_rgb_image, input_camera_intrinsics = cropping.rescale_image(
                    input_rgb_image, input_camera_intrinsics, output_resolution)
                
                assert (input_camera_intrinsics - camera_intrinsics).sum() < 1e-6
                save_img_path = os.path.join("/mydata/data/seunghoonjeong/co3dv2_filtered", filepath)
                os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)

                input_rgb_image.save(save_img_path)

                save_meta_path = save_img_path.replace('jpg', 'npz')
                np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                        camera_pose=camera_pose)
        else:
            print(f"{seq_name} is not 360 seq")

    # centric_seq_names_all[category] = centric_seq_names
    # centric_seq_info_mini_all[category] = centric_seq_info_mini
    # centric_seq_info_full_all[category] = centric_seq_info_full
    
    os.makedirs(os.path.join(filtered_dir, category), exist_ok=True)
    with open(os.path.join(filtered_dir, category, "co3dv2_centric_seq_name.json"), "w") as f:
        json.dump(centric_seq_names, f, indent=4)
        
    with open(os.path.join(filtered_dir, category, "co3dv2_centric_seq_info_mini.json"), "w") as f:
        json.dump(centric_seq_info_mini, f, indent=4)
        
    with open(os.path.join(filtered_dir, category, "co3dv2_centric_seq_info_full.json"), "w") as f:
        json.dump(centric_seq_info_full, f)
        
# with open(os.path.join(filtered_dir, "co3dv2_centric_seq_names_all.json"), "w") as f:
#     json.dump(centric_seq_names_all, f, indent=4)
    
# with open(os.path.join(filtered_dir, "co3dv2_centric_seq_info_mini_all.json"), "w") as f:
#     json.dump(centric_seq_info_mini_all, f, indent=4)
    
# with open(os.path.join(filtered_dir, "co3dv2_centric_seq_info_full_all.json"), "w") as f:
#     json.dump(centric_seq_info_full_all, f)
