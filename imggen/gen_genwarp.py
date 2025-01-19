import os

import matplotlib
import matplotlib.cm

import torch
import numpy as np
from PIL import Image
import cv2

import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image

from imggen.genwarp.ops import (
    get_projection_matrix, focal_length_to_fov
)
from imggen.genwarp import GenWarp


def load_gen_model(device):
    genwarp_cfg = dict(
        pretrained_model_path='checkpoints/genwarp',
        checkpoint_name='multi2',
        half_precision_weights=True
    )
    genwarp = GenWarp(cfg=genwarp_cfg)
    return genwarp

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def visualize_generated(renders, ref_image, ref_depth, save_path):
    src_pil = to_pil_image(ref_image[0])
    depth_pil = to_pil_image(colorize(ref_depth[0].float()))
    warped_pil = to_pil_image(renders['warped'][0])
    synthesized_pil = to_pil_image(renders['synthesized'][0])

    # Visualize.
    vis = Image.new('RGB', (512 * 4, 512 * 1))
    vis.paste(src_pil, (512 * 0, 0))
    vis.paste(depth_pil, (512 * 1, 0))
    vis.paste(warped_pil, (512 * 2, 0))
    vis.paste(synthesized_pil, (512 * 3, 0))
    vis.save(save_path)

def choose_idx_to_generate(seq_info, known_area_ratio):
    generate_cnt = 2 ** seq_info.cur_stage
    min_known_area_ratio = 0.5

    # get lowest known area ratio that is greater than min_known_area_ratio
    candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] >= min_known_area_ratio and seq_info.views[i].stage == -1]
    candidates.sort(key=lambda x: x[1])
    
    generate_idx = [candidates[i][0] for i in range(generate_cnt)]

    if len(generate_idx) < generate_cnt:
        additional_candidates = [(i, known_area_ratio[i]) for i in range(seq_info.length) if known_area_ratio[i] < min_known_area_ratio and seq_info.views[i].stage == -1]
        additional_candidates.sort(key=lambda x: -x[1])
        additional_generate_cnt = generate_cnt - len(generate_idx)
        generate_idx += [additional_candidates[i][0] for i in range(additional_generate_cnt)]

    print(f"Generate {generate_cnt} images:")
    for idx in generate_idx:
        print(f"  {idx} -> {known_area_ratio[idx]}")
    return generate_idx

def generate_images(gen_model, seq_info, generate_idx):
    for idx in generate_idx:
        overlaps = seq_info.views[idx].overlaps
        # get biggest overlap
        reference_idx = overlaps.index(max(overlaps))
        print(f"Generate image {idx} from reference {reference_idx} ({overlaps[reference_idx]})")

        reference_view = seq_info.views[reference_idx]
        ref_idx = reference_view.ref_idx
        ref_image = reference_view.image.detach().cpu().numpy()
        ref_image = ((ref_image + 1) / 2 * 255).astype(np.uint8)
        # ref_image = to_tensor(Image.fromarray(cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)))[None].to(seq_info.device)
        ref_image = to_tensor(Image.fromarray(ref_image))[None].to(seq_info.device)
        ref_image = ref_image.half()

        ref_pts = np.load(os.path.join(seq_info.project_dir, "recon", f"pts3d_{ref_idx}.npy"))
        ref_pts = ref_pts.reshape(-1, 3)
        ref_pts = torch.tensor(ref_pts).float().to(seq_info.device)

        ref_pose = reference_view.pose
        ref_R = ref_pose[:3, :3]
        ref_T = ref_pose[:3, 3]
        ref_K = reference_view.get_intrinsic_matrix()
        ref_proj = ((ref_pts - ref_T) @ ref_R) @ ref_K.T
        ref_depth = ref_proj[:, 2].reshape(reference_view.size[0], reference_view.size[1])
        ref_depth = ref_depth.unsqueeze(0).unsqueeze(0).half()

        ref_focal = reference_view.focal
        ref_h = reference_view.size[0]
        ref_fov = focal_length_to_fov(ref_focal, ref_h)
        ref_fov = torch.ones(1) * ref_fov
        near, far = 0.01, 100
        src_proj_mtx = get_projection_matrix(
            fovy=ref_fov, 
            aspect_wh=reference_view.size[1] / reference_view.size[0],
            near=near,
            far=far
        ).to(seq_info.device).half()
        
        target_view = seq_info.views[idx]
        tar_focal = target_view.focal
        tar_h = target_view.size[0]
        tar_fov = focal_length_to_fov(tar_focal, tar_h)
        tar_fov = torch.ones(1) * tar_fov
        tar_proj_mtx = get_projection_matrix(
            fovy=tar_fov,
            aspect_wh=target_view.size[1] / target_view.size[0],
            near=near,
            far=far
        ).to(seq_info.device).half()

        tar_pose = target_view.pose
        
        rel_view_mtx = (tar_pose @ torch.linalg.inv(ref_pose)).to(seq_info.device).unsqueeze(0).half()

        tar_warped = target_view.warped.half()
        tar_mask = target_view.mask

        save_path = os.path.join(seq_info.project_dir, "recon", f"{idx}.png")

        breakpoint()

        renders = gen_model(
            src_image=ref_image,
            src_depth=ref_depth,
            rel_view_mtx=rel_view_mtx,
            src_proj_mtx=src_proj_mtx,
            tar_proj_mtx=tar_proj_mtx,
        )

        visualize_generated(renders, ref_image, ref_depth, save_path)

        # renders = gen_model.gen_with_warped(
        #     src_image=ref_image,
        #     src_depth=ref_depth,
        #     rel_view_mtx=rel_view_mtx,
        #     src_proj_mtx=src_proj_mtx,
        #     tar_proj_mtx=tar_proj_mtx,
        #     warped=tar_warped,
        #     mask=tar_mask
        # )

        generated_image = renders['synthesized']
        generated_image = to_pil_image(generated_image[0])

        generated_image.save(os.path.join(seq_info.project_dir, "recon", f"{idx}.png"))
        
