import os
import json
import torch

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from util.util_pose import generate_smooth_camera_path
from util.extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer

def get_inputs(input_dir):
    extrinsics = None
    intrinsics = None
    trajectory = None
    reference = None

    # Load images
    input_dir_list = os.listdir(input_dir)
    image_paths = []
    for file_name in input_dir_list:
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            image_paths.append(os.path.join(input_dir, file_name))
    image_paths.sort()
    print(image_paths)
    assert len(image_paths) > 1, "Number of images should be more than 1."

    images = []
    for image_path in image_paths:
        images.append(imageio.imread(image_path))
    images = np.stack(images)
    images = torch.tensor(images).float() / 127.5 - 1.0

    # If image is rgba, remove alpha channel (4 -> 3)
    if images.shape[3] == 4:
        images = images[:,:,:,:3]

    # Resize to (2, 512, 512, 3)
    images = torch.nn.functional.interpolate(images.permute(0,3,1,2), size=(512, 512), mode='bilinear', align_corners=False)
    images = images.permute(0,2,3,1)

    # Load extrinsics, intrinsics, and trajectory
    if os.path.exists(os.path.join(input_dir, "extrinsics.json")):
        with open(os.path.join(input_dir, "extrinsics.json"), "r") as f:
            data_extrinsics = json.load(f)

        assert len(data_extrinsics["extrinsics"]) == len(image_paths), f"Number of extrinsics {len(data_extrinsics['extrinsics'])} does not match the number of images {len(image_paths)}."

        extrinsics = torch.tensor(data_extrinsics["extrinsics"]).float()
        

    if os.path.exists(os.path.join(input_dir, "intrinsics.json")):
        with open(os.path.join(input_dir, "intrinsics.json"), "r") as f:
            data_intrinsics = json.load(f)

        assert len(data_intrinsics["focals"]) == len(image_paths), f"Number of focals {len(data_intrinsics['focals'])} does not match the number of images {len(image_paths)}."
        assert len(data_intrinsics["principal_points"]) == len(image_paths), f"Number of principal points {len(data_intrinsics['principal_points'])} does not match the number of images {len(image_paths)}."
        
        intrinsics = {
            "focals": data_intrinsics["focals"],
            "principal_points": torch.tensor(data_intrinsics["principal_points"]).float()
        }

    if os.path.exists(os.path.join(input_dir, "trajectory.json")):
        with open(os.path.join(input_dir, "trajectory.json"), "r") as f:
            data_trajectory = json.load(f)

        trajectory = torch.tensor(data_trajectory["trajectory"]).float()
        reference = data_trajectory["reference"]

        if extrinsics is not None:
            for i in range(len(reference)):
                if reference[i] == -1:
                    continue

                assert torch.allclose(trajectory[i], extrinsics[reference[i]]), f"Trajectory does not match the extrinsics at index {i}."

        else:
            print("Extrinsics do not exist. Will get extrinsics from the trajectory.")
            for i in range(len(reference)):
                if reference[i] == -1:
                    continue

                if extrinsics is None:
                    extrinsics = trajectory[reference[i]].unsqueeze(0)
                else:
                    extrinsics = torch.cat((extrinsics, trajectory[reference[i]].unsqueeze(0)), dim=0)

                assert len(extrinsics) == len(image_paths), f"Number of extrinsics {len(extrinsics)} does not match the number of images {len(image_paths)}."

            data_extrinsics = {
                "extrinsics": extrinsics.detach().cpu().numpy().tolist()
            }
            with open(os.path.join(input_dir, "extrinsics.json"), "w") as f:
                json.dump(data_extrinsics, f)
            

    return image_paths, images, extrinsics, intrinsics, trajectory, reference

def get_trajectory(extrinsics, input_dir, length=20):
    extrinsics = extrinsics.detach().cpu().numpy()
    trajectory = np.zeros((length, 4, 4))
    reference = [-1] * length
    
    given_pose_cnt = len(extrinsics)
    given_pose_idx = [int(i * (length - 1) / (given_pose_cnt - 1)) for i in range(given_pose_cnt-1)] + [length-1]
    print(f"Given pose cnt : {given_pose_cnt}, Given pose idx : {given_pose_idx}")

    for i, idx in enumerate(given_pose_idx):
        reference[idx] = i

    for i in range(given_pose_cnt-1):
        start_pose = extrinsics[i]
        end_pose = extrinsics[i+1]
        positions, rotations = generate_smooth_camera_path(start_pose[:3,:3], start_pose[:3,3], end_pose[:3,:3], end_pose[:3,3], given_pose_idx[i+1] - given_pose_idx[i])
        for j in range(given_pose_idx[i], given_pose_idx[i+1]):
            R = np.array(rotations[j - given_pose_idx[i]])
            T = np.array(positions[j - given_pose_idx[i]]).reshape(-1,1)
            matrix = np.concatenate((np.concatenate((R, T), axis=1), np.array([[0,0,0,1]])), axis=0)
            trajectory[j] = matrix
    trajectory[-1] = extrinsics[-1]

    print("Trajectory do not exist. Will save the generated trajectory.")
    with open(os.path.join(input_dir, "trajectory.json"), "w") as f:
        json.dump({"trajectory": trajectory.tolist(), "reference": reference}, f)

    trajectory = torch.tensor(trajectory).float()
    return trajectory, reference
    
def visualize_pose(extrinsics, trajectory, save_dir):
    visualizer_ext = CameraPoseVisualizer([-3, 3], [-3, 3], [0, 3])
    for i in range(len(extrinsics)):
        visualizer_ext.extrinsic2pyramid(extrinsics[i].detach().cpu().numpy(), plt.cm.rainbow(i / len(extrinsics)), focal_len_scaled=1)
    plt.title('Extrinsics')
    plt.savefig(os.path.join(save_dir, "vis_extrinsics.png"))

    visualizer_trj = CameraPoseVisualizer([-3, 3], [-3, 3], [0, 3])
    for i in range(len(trajectory)):
        visualizer_trj.extrinsic2pyramid(trajectory[i].detach().cpu().numpy(), plt.cm.rainbow(i / len(trajectory)), focal_len_scaled=1)
    plt.title('Trajectory')
    plt.savefig(os.path.join(save_dir, "vis_trajectory.png"))