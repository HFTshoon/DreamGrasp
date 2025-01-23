import argparse
import os
import json

from PIL import Image
import numpy as np
import open3d as o3d

from util.util_preprocess import make_pts3d

def main(args):
    data_dir = args.data_dir
    category_name = data_dir.split("/")[-2]
    seq_name = data_dir.split("/")[-1]

    seq_path = os.path.join(data_dir, f"../../selected_seqs_{args.split}.json")
    with open(seq_path, "r") as f:
        seq_info = json.load(f)
    seq = seq_info[category_name][seq_name]

    if args.idx is None:
        idx = [i for i in range(len(depthmaps))]
    else:
        str_idx = args.idx.split(",")
        idx = [int(i) for i in str_idx]

    image_paths = []
    for i in idx:
        image_paths.append(os.path.join(data_dir, "images", f"frame{seq[i]:06d}.jpg"))
    print(f"Data directory: {data_dir}, split: {args.split}")
    print(f"Image directory: {' '.join(image_paths)}")

    depthmaps = np.load(os.path.join(data_dir, f"depthmaps_{args.split}.npy"))
    focals = np.load(os.path.join(data_dir, f"focals_{args.split}.npy"))
    pps = np.load(os.path.join(data_dir, f"pps_{args.split}.npy"))
    poses = np.load(os.path.join(data_dir, f"poses_{args.split}.npy"))
    pts3d = make_pts3d(depthmaps, poses, focals, pps)
    h,w = depthmaps[0].shape    

    all_things = []
    for i, image_path in zip(idx, image_paths):
        image = Image.open(image_path)
        pts = pts3d[i].reshape(-1,3)
        colors = np.array(image).reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors/255)
        all_things.append(pcd)

        pose = poses[i]
        T = pose[:3,3]
        # make sphere at camera center
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 0.1, 0.7])
        sphere.translate(T)
        all_things.append(sphere)
    o3d.visualization.draw_geometries(all_things)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", "-d", required=True, type=str)
    arg_parser.add_argument("--split", "-s", type=str, default="train")
    arg_parser.add_argument("--idx", "-i", type=str)
    args = arg_parser.parse_args()

    main(args)