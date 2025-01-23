import os
import json
import numpy as np

import torch

# add sys path for dust3r
import sys
sys.path.append("recon")
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def load_recon_model(use_mast3r, device):
    if use_mast3r:
        from mast3r.model import AsymmetricMASt3R as ReconModel
        model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    else:
        from dust3r.model import AsymmetricCroCo3DStereo as ReconModel
        model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

    model = ReconModel.from_pretrained(model_path).to(device)
    return model

def recon_3d_init(recon_model, device, image_paths, extrinsics, intrinsics, input_dir, output_dir, use_mast3r):
    os.makedirs(os.path.join(output_dir, "recon"), exist_ok=True)

    image_cnt = len(image_paths)
    if extrinsics is not None and intrinsics is not None:
        has_recon_result = True
        for i in range(image_cnt):
            if not os.path.exists(os.path.join(output_dir, "recon", f"pts3d_{i}.npy")):
                has_recon_result = False
                break
        if has_recon_result:
            print("Skip initial reconstruction. Will use the existing recon results.")
            return extrinsics, intrinsics

    images = load_images(image_paths, size=512, square_ok=True)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, recon_model, device, batch_size=1)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    if use_mast3r:
        scene.min_conf_thr = 1.5

    if extrinsics is not None:
        scene.preset_pose(extrinsics)

    if intrinsics is not None:
        scene.preset_focal(intrinsics["focals"])
        scene.preset_principal_point(intrinsics["principal_points"])
    else:
        scene.preset_principal_point_zero()

    loss = scene.compute_global_alignment(init="known_poses", niter=300, schedule="cosine", lr=0.01)
    
    # (H,W) = (512, 512) if square_ok=True, (384, 512) if square_ok=False
    pts3d = scene.get_pts3d()               # B X (H, W, 3)
    focals = scene.get_focals()             # (B, 1)
    poses = scene.get_im_poses()            # (B, 4, 4)
    pps = scene.get_principal_points()      # (B, 2)
    # imgs = scene.imgs                       # (B, H, W, 3)
    # confidence_masks = scene.get_masks()    # B X (H, W)
    # conf = scene.get_conf()                 # B X (H, W)
    depthmaps = scene.get_depthmaps()       # B X (H, W)

    for i in range(image_cnt):
        pts3d_i = pts3d[i].detach().cpu().numpy()
        np.save(os.path.join(output_dir, "recon", f"pts3d_{i}.npy"), pts3d_i)
        depthmaps_i = depthmaps[i].detach().cpu().numpy()
        np.save(os.path.join(output_dir, "recon", f"depthmap_{i}.npy"), depthmaps_i)

    if extrinsics is None:
        extrinsics = poses
        print("Extrinsics do not exist. Will save the recon model output.")
        data_extrinsics = {
            "extrinsics": poses.detach().cpu().numpy().tolist()
        }
        with open(os.path.join(input_dir, 'extrinsics.json'), 'w') as f:
            json.dump(data_extrinsics, f)

    if intrinsics is None:
        intrinsics = {
            "focals": focals,
            "principal_points": pps
        }
        print("Intrinsics do not exist. Will save the recon model output.")
        data_intrinsics = {
            "focals": [focal[0] for focal in focals.detach().cpu().numpy().tolist()],
            "principal_points": pps.detach().cpu().numpy().tolist()
        }
        with open(os.path.join(input_dir, 'intrinsics.json'), 'w') as f:
            json.dump(data_intrinsics, f)

    return extrinsics, intrinsics

def recon_3d_incremental(recon_model, device, seq_info, generate_idx, use_mast3r):
    seq_dir = os.path.join(seq_info.project_dir, "sequence")
    seq_dir_list = os.listdir(seq_dir)
    image_paths = []
    for file_name in seq_dir_list:
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            image_paths.append(os.path.join(seq_dir, file_name))
    image_paths.sort()
    print(image_paths)
    assert len(image_paths) > 1, "Number of images should be more than 1."

    newly_generated = [False] * len(image_paths)
    poses = []
    focals = []
    pps = []
    depthmaps = []
    for idx, image_path in enumerate(image_paths):
        image_file_name = image_path.split("/")[-1]
        image_idx = int(image_file_name.split(".")[0])
        poses.append(seq_info.views[image_idx].pose)
        focals.append(seq_info.views[image_idx].focal)
        pps.append(seq_info.views[image_idx].pp)

        ref_idx = seq_info.views[image_idx].ref_idx
        assert ref_idx != -1, "Reference index should not be -1."
        if image_idx in generate_idx:
            newly_generated[idx] = True
        else:
            depthmap = np.load(os.path.join(seq_info.project_dir, "recon", f"depthmap_{ref_idx}.npy"))
            depthmaps.append(torch.tensor(depthmap).to(device))
    
    input_poses = torch.stack(poses, dim=0).float().to(device)
    input_focals = focals
    input_pps = torch.stack(pps, dim=0).float().to(device)

    depthmap_msk = [not newly_generated[i] for i in range(len(image_paths))]

    images = load_images(image_paths, size=512, square_ok=True)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True, new_data_msk=newly_generated)
    output = inference(pairs, recon_model, device, batch_size=1)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    if use_mast3r:
        scene.min_conf_thr = 1.5

    print(depthmap_msk)
    scene.preset_pose(input_poses)
    scene.preset_focal(input_focals)
    scene.preset_principal_point(input_pps)
    scene.preset_depthmap(depthmaps, depthmap_msk)

    loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
    
    # (H,W) = (512, 512) if square_ok=True, (384, 512) if square_ok=False
    pts3d = scene.get_pts3d()               # B X (H, W, 3)
    focals = scene.get_focals()             # (B, 1)
    poses = scene.get_im_poses()            # (B, 4, 4)
    pps = scene.get_principal_points()      # (B, 2)
    # imgs = scene.imgs                       # (B, H, W, 3)
    # confidence_masks = scene.get_masks()    # B X (H, W)
    # conf = scene.get_conf()                 # B X (H, W)
    depthmaps = scene.get_depthmaps()       # B X (H, W)

    for idx, image_path in enumerate(image_paths):
        image_file_name = image_path.split("/")[-1]
        image_idx = int(image_file_name.split(".")[0])

        ref_idx = seq_info.views[image_idx].ref_idx
        assert ref_idx != -1, "Reference index should not be -1."
        if image_idx in generate_idx:
            pts3d_i = pts3d[idx].detach().cpu().numpy()
            np.save(os.path.join(seq_info.project_dir, "recon", f"pts3d_{ref_idx}.npy"), pts3d_i)
            depthmaps_i = depthmaps[idx].detach().cpu().numpy()
            np.save(os.path.join(seq_info.project_dir, "recon", f"depthmap_{ref_idx}.npy"), depthmaps_i)