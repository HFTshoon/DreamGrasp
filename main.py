import os
import argparse

import torch

from recon.recon_3d import recon_3d_init, load_recon_model, recon_3d_incremental
from imggen.gen_zeronvs import generate_images, choose_idx_to_generate, load_gen_model
from util.seq_info import SeqInfo
from util.util_data import get_inputs, get_trajectory, visualize_pose
from util.util_warp import warp_reference

def main(args, device):
    if args.input_dir is None:
        args.input_dir = f"assets/{args.exp_name}"

    if args.output_dir is None:
        args.output_dir = f"results/{args.exp_name}"
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_dir is None:
        args.model_dir = f"imggen/megascenes/train_results/{args.exp_name}"

    print(f"Experiment name: {args.exp_name}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    recon_model = load_recon_model(args.use_mast3r, device)

    gen_model = load_gen_model(args.output_dir, args.model_dir, args.model_iteration, device)

    image_paths, images, extrinsics, intrinsics, trajectory, reference = get_inputs(args.input_dir, device)

    extrinsics, intrinsics = recon_3d_init(recon_model, device, image_paths, extrinsics, intrinsics, args.input_dir, args.output_dir, args.use_mast3r)
    
    if trajectory is None:
        print("Trajectory does not exist. Will generate a smooth camera path.")
        trajectory, reference = get_trajectory(extrinsics, args.input_dir)
        trajectory = trajectory.to(device)

    visualize_pose(extrinsics, trajectory, args.output_dir)

    seq_info = SeqInfo(images, extrinsics, intrinsics, trajectory, reference, args.output_dir, device)

    for i in range(seq_info.required_stage):
        print("Stage", seq_info.cur_stage)
        known_area_ratio = warp_reference(seq_info)
        generate_idx = choose_idx_to_generate(seq_info, known_area_ratio)
        generate_images(gen_model, seq_info, generate_idx)
        recon_3d_incremental(recon_model, device, seq_info, generate_idx, args.use_mast3r)
        seq_info.cur_stage += 1



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_name", "-n", required=True, type=str)
    arg_parser.add_argument("--input_dir", "-i", type=str)
    arg_parser.add_argument("--output_dir", "-o", type=str)
    arg_parser.add_argument("--use_mast3r", "-m", action="store_true")
    arg_parser.add_argument("--model_dir", "-c", type=str)
    arg_parser.add_argument("--model_iteration", "-it", default=None, type=int)
    args = arg_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args, device)