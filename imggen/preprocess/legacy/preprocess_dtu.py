import os
import os.path as osp
import argparse
import json
import random

import numpy as np
import PIL.Image

import util.util_preprocess as cropping  # noqa

from recon.mast3r.model import AsymmetricMASt3R

from recon.dust3r.inference import inference
from recon.dust3r.utils.image import load_images
from recon.dust3r.image_pairs import make_pairs
from recon.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/mydata/data/seunghoonjeong/DTU_single_preprocess')
    parser.add_argument("--dtu_dir", type=str, default='/mydata/data/seunghoonjeong/DTU_single')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan", type=int, default=0)
    parser.add_argument("--num_sequences_per_scan", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser

def prepare_sequences(model, scan, dtu_dir, output_dir, img_size, split, max_num_sequences_per_scan, seed):
    recon_frame_cnt = 8
    random.seed(seed)
    scan_dir = osp.join(dtu_dir, scan)
    scan_output_dir = osp.join(output_dir, scan)

    image_list = [f for f in os.listdir(osp.join(scan_dir, 'image')) if f.endswith('.png')]
    image_list.sort()

    selected_sequences_numbers_dict = {}
    for cur_seq_num in range(max_num_sequences_per_scan):
        if split == 'test':
            seq_num = cur_seq_num + max_num_sequences_per_scan
        else:
            seq_num = cur_seq_num

        sequence_image_list = random.sample(image_list, recon_frame_cnt)
        sequence_image_num_list = [int(image_name.split('.')[0]) for image_name in sequence_image_list]
        selected_sequences_numbers_dict[f"seq{seq_num:03d}"] = sequence_image_num_list

    error_count = 0
    error_sequences = []
    for cur_seq_num in range(max_num_sequences_per_scan):
        if split == 'test':
            seq_num = cur_seq_num + max_num_sequences_per_scan
        else:
            seq_num = cur_seq_num

        original_seq_path = os.path.join(scan_dir, 'image')
        seq_path = os.path.join(scan_output_dir, f"seq{seq_num:03d}", "images")
        os.makedirs(seq_path, exist_ok=True)
        seq_list = os.listdir(seq_path)
        image_list = [f"{frame_idx:06d}.png" for frame_idx in selected_sequences_numbers_dict[f"seq{seq_num:03d}"]]
        image_list.sort()

        for image_name in image_list:
            original_image_path = os.path.join(original_seq_path, image_name)
            save_image_path = os.path.join(seq_path, image_name)
            input_rgb_image = PIL.Image.open(original_image_path).convert('RGB')

            W, H = input_rgb_image.size

            cx, cy = W // 2, H // 2
            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)
            min_margin = min(min_margin_x, min_margin_y)

            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = cx - min_margin, cy - min_margin
            r, b = cx + min_margin, cy + min_margin
            crop_bbox = (l, t, r, b)
            image = input_rgb_image.crop(crop_bbox)

            output_resolution = np.array([img_size, img_size])
            input_resolution = np.array(image.size)
            scale_final = max(output_resolution / input_resolution) + 1e-8
            output_resolution = np.floor(input_resolution * scale_final).astype(int)
            image = image.resize(tuple(output_resolution), resample=PIL.Image.LANCZOS if scale_final < 1 else PIL.Image.BICUBIC)
            image.save(save_image_path)

        assert len(image_list) == recon_frame_cnt, f"len(image_list) != {recon_frame_cnt}: {len(image_list)} {seq_path}"

        img_path_list = [os.path.join(seq_path, image_name) for image_name in image_list]
        images = load_images(img_path_list, size=512, square_ok=True)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        try:
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            scene.min_conf_thr = 1.5 # for MASt3R
            
            scene.preset_principal_point_zero()

            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            
            poses = scene.get_im_poses()            # (10, 4, 4)
            focals = scene.get_focals()             # (10, 1)
            pps = scene.get_principal_points()      # (10, 2)
            depthmaps = scene.get_depthmaps()       # 10 x (512, 512)

            poses = poses.detach().cpu().numpy()
            focals = focals.detach().cpu().numpy()
        except:
            # delete this seq from selected_sequences_numbers_subset_dict
            print(f"Error in seq{seq_num:03d}")
            del selected_sequences_numbers_dict[f"seq{seq_num:03d}"]
            error_count += 1
            error_sequences.append(os.path.join(scan_output_dir, f"seq{seq_num:03d}"))
            continue

        depthmaps_numpy = np.zeros((recon_frame_cnt, 512, 512), dtype=np.float32)
        for i, dps in enumerate(depthmaps):
            dps = dps.detach().cpu().numpy()
            depthmaps_numpy[i] = dps

        pps = pps.detach().cpu().numpy()

        np.save(os.path.join(scan_output_dir, f"seq{seq_num:03d}", f"poses_{split}.npy"), poses)
        np.save(os.path.join(scan_output_dir, f"seq{seq_num:03d}", f"focals_{split}.npy"), focals)
        np.save(os.path.join(scan_output_dir, f"seq{seq_num:03d}", f"depthmaps_{split}.npy"), depthmaps_numpy)
        np.save(os.path.join(scan_output_dir, f"seq{seq_num:03d}", f"pps_{split}.npy"), pps)
        # np.savez(os.path.join(scan_output_dir, f"seq{seq_num:03d}", f'recon.npz'), poses=poses, focals=focals, pts3d=pts3d_numpy, pps=pps)

    print(f"Error count: {error_count}")
    print(f"Error sequences: {error_sequences}")
    return selected_sequences_numbers_dict

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.dtu_dir != args.output_dir
    if args.scan != 0:
        scans = [f"scan{args.scan}"]
    else:
        scans = [dir_name for dir_name in os.listdir(args.dtu_dir) if dir_name.startswith('scan')]

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    device = 'cuda'
    batch_size = 28
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    for split in ['train', 'test']:
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path):
            continue

        all_selected_sequences = {}
        for scan in scans:
            scan_output_dir = osp.join(args.output_dir, scan)
            os.makedirs(scan_output_dir, exist_ok=True)
            scan_selected_sequences_path = os.path.join(scan_output_dir, f'selected_seqs_{split}.json')
            if os.path.isfile(scan_selected_sequences_path):
                print(f"Skipping {split} - {scan}")
                with open(scan_selected_sequences_path, 'r') as fid:
                    scan_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - {scan}")
                scan_selected_sequences = prepare_sequences(
                    model=model,
                    scan=scan,
                    dtu_dir=args.dtu_dir,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    max_num_sequences_per_scan=args.num_sequences_per_scan,
                    seed=args.seed + int(scan.split('scan')[1]) + (0 if split == 'train' else 1000)
                )
                with open(scan_selected_sequences_path, 'w') as file:
                    json.dump(scan_selected_sequences, file)

            all_selected_sequences[scan] = scan_selected_sequences
        with open(selected_sequences_path, 'w') as file:
            json.dump(all_selected_sequences, file)