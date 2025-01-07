import os
import argparse

from util.util_data import get_inputs, get_trajectory, visualize_pose
from util.util_warp import warp_reference
from recon.recon_3d import recon_3d_init, load_recon_model

def main(args):
    if args.input_dir is None:
        args.input_dir = f"assets/{args.exp_name}"

    if args.output_dir is None:
        args.output_dir = f"results/{args.exp_name}"
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Experiment name: {args.exp_name}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    recon_model = load_recon_model(args.use_mast3r)

    image_paths, images, extrinsics, intrinsics, trajectory, reference = get_inputs(args.input_dir)

    extrinsics, intrinsics = recon_3d_init(recon_model, image_paths, extrinsics, intrinsics, args.input_dir, args.output_dir, args.use_mast3r)
    
    if trajectory is None:
        print("Trajectory does not exist. Will generate a smooth camera path.")
        trajectory, reference = get_trajectory(extrinsics, args.input_dir)

    visualize_pose(extrinsics, trajectory, args.output_dir)

    target_poses_cnt = 0
    for ref in reference:
        if not ref:
            target_poses_cnt += 1
    print(f"Number of target poses: {target_poses_cnt}")

    #for i in range(target_poses_cnt):
    warp_reference(images, extrinsics, intrinsics, trajectory, reference, args.output_dir)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_name", "-n", required=True, type=str)
    arg_parser.add_argument("--input_dir", "-i", type=str)
    arg_parser.add_argument("--output_dir", "-o", type=str)
    arg_parser.add_argument("--use_mast3r", "-m", action="store_true")
    args = arg_parser.parse_args()

    main(args)