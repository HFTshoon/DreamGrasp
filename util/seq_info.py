import os

import numpy as np
import torch
from PIL import Image

from util.util_pose import get_cameras_origin, get_angle_between_pair_cameras
from util.util_warp import compute_optical_flow, get_warped_coords

class ViewInfo():
    def __init__(self, image, ref_idx, size, pose, focal, pp, stage, length, device):
        self.image = image
        self.ref_idx = ref_idx
        self.size = size
        self.pose = pose
        self.focal = focal
        self.pp = pp
        self.stage = stage
        self.device = device
        self.warped = None
        self.mask = None
        self.overlaps = [0] * length

    def get_intrinsic_matrix(self):
        return torch.tensor([
            [self.focal, 0, self.pp[0]], 
            [0, self.focal, self.pp[1]], 
            [0, 0, 1]
        ]).float().to(self.device)
    
    def set_warped_and_mask(self, warped, mask):
        self.warped = warped
        self.mask = mask

    def set_overlap_with_reference(self, reference_idx, overlap_value):
        self.overlaps[reference_idx] = overlap_value

class SeqInfo():
    def __init__(self, images, extrinsics, intrinsics, trajectory, reference, project_dir, device):
        self.device = device
        self.project_dir = project_dir
        os.makedirs(os.path.join(project_dir, "sequence"), exist_ok=True)

        self.length = len(trajectory)
        self.views = [None for _ in range(self.length)]
        
        self.default_focal = intrinsics["focals"][0]
        self.default_pp = intrinsics["principal_points"][0]
        self.default_h = images[0].shape[0]
        self.default_w = images[0].shape[1]

        self.reference = reference

        target_poses_cnt = 0
        reference_poses_cnt = 0
        for i in range(self.length):
            if reference[i] == -1:
                self.views[i] = ViewInfo(
                    image = None, # Have to generate this
                    ref_idx = -1, # No reference
                    size = (self.default_h, self.default_w), 
                    pose = trajectory[i], 
                    focal = self.default_focal, 
                    pp = self.default_pp, 
                    stage = -1, # Not a reference view yet
                    length = self.length,
                    device = device
                )
                target_poses_cnt += 1

            else:
                assert torch.allclose(trajectory[i], extrinsics[reference[i]]), f"Pose does not match at index {i}: {trajectory[i]}, {extrinsics[reference[i]]}"
                image = (images[reference[i]].cpu().detach().numpy() + 1) * 127.5
                image = image.astype(np.uint8)
                Image.fromarray(image).save(os.path.join(project_dir, "sequence", f"{i}.png"))

                self.views[i] = ViewInfo(
                    image = images[reference[i]], 
                    ref_idx = reference[i],
                    size = (images[reference[i]].shape[0], images[reference[i]].shape[1]),
                    pose = extrinsics[reference[i]], 
                    focal = intrinsics["focals"][reference[i]], 
                    pp = intrinsics["principal_points"][reference[i]], 
                    stage = 0,
                    length = self.length,
                    device = device
                )
                reference_poses_cnt += 1

        self.target_poses_cnt = target_poses_cnt
        self.reference_poses_cnt = reference_poses_cnt
        self.generated_poses_cnt = 0

        Rs = [pose[:3, :3].detach().cpu().numpy() for pose in trajectory]
        Ts = [pose[:3, 3].detach().cpu().numpy() for pose in trajectory]
        self.scene_origin = get_cameras_origin(Rs, Ts)
        print(f"Scene origin: {self.scene_origin}")

        self.pair_angles = np.zeros((self.length, self.length))
        for i in range(self.length):
            for j in range(self.length):
                if i == j:
                    continue
                self.pair_angles[i, j] = get_angle_between_pair_cameras(Ts[i], Ts[j], self.scene_origin)

        required_stage = 0
        while target_poses_cnt > 1:
            target_poses_cnt //= 2
            required_stage += 1
        required_stage += 1

        self.required_stage = required_stage
        print(f"Target poses count: {self.target_poses_cnt}, Required stage: {self.required_stage}")

        self.cur_stage = 0

    def get_flow_query_from_reference(self, query_idx, reference_idx):
        query_view = self.views[query_idx]
        reference_view = self.views[reference_idx]
        assert reference_view.stage != -1, f"View number {reference_idx} is not a reference view"

        query_R = query_view.pose[:3, :3]
        query_T = query_view.pose[:3, 3]
        query_K = query_view.get_intrinsic_matrix()
        reference_R = reference_view.pose[:3, :3]
        reference_T = reference_view.pose[:3, 3]
        reference_K = reference_view.get_intrinsic_matrix()

        reference_h, reference_w = reference_view.size
        reference_pts = np.load(os.path.join(self.project_dir, "recon", f"pts3d_{reference_view.ref_idx}.npy"))
        reference_pts = reference_pts.reshape(-1, 3)
        reference_pts = torch.tensor(reference_pts).float().to(self.device)

        flow, depth = compute_optical_flow(reference_pts, reference_R, reference_T, reference_K, query_R, query_T, query_K)
        flow = flow.reshape(reference_h, reference_w, 2)

        return flow, depth
    
    def get_warped_coords_query_from_reference(self, query_idx, reference_idx):
        grid = torch.stack(torch.meshgrid(torch.arange(0, self.default_h), torch.arange(0, self.default_w)), dim=-1).float()
        coords = torch.stack((grid[..., 0]/self.default_h, grid[..., 1]/self.default_w), dim=-1).to(self.device) # (H, W, 2)
        freq_bands = 2.**torch.linspace(0., 1., steps=2)
        peridoic_funcs = [torch.sin, torch.cos]
        embed_fns = []
        for freq in freq_bands:
            for p_fn in peridoic_funcs:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))

        coords = torch.cat([fn(coords) for fn in embed_fns], dim=-1) # (H, W, 8)

        flow, depth = self.get_flow_query_from_reference(query_idx, reference_idx)
        warped_coords = get_warped_coords(coords, flow, depth)
        return coords, warped_coords
    
    def set_generated_image(self, idx, image):
        self.views[idx].image = image
        self.views[idx].stage = self.cur_stage + 1
        self.views[idx].ref_idx = self.reference_poses_cnt
        self.reference[idx] = self.reference_poses_cnt
        self.reference_poses_cnt += 1
        self.generated_poses_cnt += 1