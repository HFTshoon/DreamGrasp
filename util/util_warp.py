import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from einops import rearrange
from splatting import splatting_function

def project_points(P, R, T, K):
    """
    3D 포인트를 2D 이미지 좌표로 투영합니다.
    
    Parameters:
    - P: (N, 3) 3D 포인트 클라우드
    - R: (3, 3) 회전 행렬
    - T: (3,) 이동 벡터
    - K: (3, 3) 카메라 내부 파라미터 행렬
    
    Returns:
    - pts_2d: (N, 2) 2D 이미지 좌표
    """
    # 카메라 좌표계로 변환
    P_cam = (P - T) @ R # (N, 3)

    # 투영 (핀홀 카메라 모델)
    P_proj = P_cam @ K.T  # (N, 3)

    depth = P_proj[:, 2]

    # 정규화
    pts_2d = P_proj[:, :2] / P_proj[:, 2, np.newaxis]
    
    return pts_2d, depth

def compute_optical_flow(P, R_source, T_source, K_source, R_query, T_query, K_query):
    """
    소스 뷰에서 쿼리 뷰로의 Optical Flow를 계산합니다.
    
    Parameters:
    - P: (N, 3) 3D 포인트 클라우드
    - R_source: (3, 3) 소스 뷰의 회전 행렬
    - T_source: (3,) 소스 뷰의 이동 벡터
    - R_query: (3, 3) 쿼리 뷰의 회전 행렬
    - T_query: (3,) 쿼리 뷰의 이동 벡터
    - K: (3, 3) 카메라 내부 파라미터 행렬
    
    Returns:
    - flow: (N, 2) Optical Flow (dx, dy)
    """
    # 소스 뷰에서의 2D 투영
    pts_source, _ = project_points(P, R_source, T_source, K_source)  # (N, 2)
    
    # 쿼리 뷰에서의 2D 투영
    pts_query, depth_query = project_points(P, R_query, T_query, K_query)  # (N, 2)
    
    # pts_source_numpy = pts_source.detach().cpu().numpy()
    # pts_query_numpy = pts_query.detach().cpu().numpy()
    # for i in range(10):
    #     x = round(pts_source_numpy[i][0],2)
    #     y = round(pts_source_numpy[i][1],2)
    #     target_x = i
    #     target_y = 0
    #     print(f"Projection : {x}, {y} -> {target_x}, {target_y}")
    #     if abs(x - target_x) > 5 or abs(y - target_y) > 5:
    #         print(f"Projection warning! : {x}, {y} -> {target_x}, {target_y}")

    # idx = np.random.choice(len(pts_source_numpy), 10, replace=False)
    # for i in idx:
    #     print(f"({i%512}, {i//512}) ",round(pts_source_numpy[i][0],2), round(pts_source_numpy[i][1],2), "->", round(pts_query_numpy[i][0],2), round(pts_query_numpy[i][1],2))

    # Optical Flow 계산
    flow = pts_query - pts_source  # (N, 2)
    
    return flow, depth_query

def warp_reference(seq_info):
    os.makedirs(os.path.join(seq_info.project_dir, f"warp_stage{seq_info.cur_stage}"), exist_ok=True)
    os.makedirs(os.path.join(seq_info.project_dir, f"mask_stage{seq_info.cur_stage}"), exist_ok=True)

    h,w = seq_info.default_h, seq_info.default_w

    known_area_ratio = [0] * seq_info.length
    for query_idx in range(seq_info.length):
        if seq_info.reference[query_idx] != -1:
            known_area_ratio[query_idx] = 1
    
        ref_images = []
        ref_flows = []
        ref_depths = []
        for idx in range(seq_info.length):
            if query_idx == idx:
                continue
            
            if seq_info.reference[idx] == -1:
                continue
            

            ref_image = seq_info.views[idx].image
            flow, depth = seq_info.get_flow_query_from_reference(query_idx, idx)

            image = rearrange(ref_image, 'h w c -> c h w').unsqueeze(0)
            ref_images.append(image)
            flow = rearrange(flow, 'h w c -> c h w').unsqueeze(0)
            ref_flows.append(flow)
            ref_depths.append(depth)

            # importance weight based on depth
            importance = 0.5 / depth
            importance -= importance.min()
            importance /= importance.max() + 1e-6
            importance = importance * 10 - 10
            importance = importance.reshape(h, w, 1)
            importance = rearrange(importance, 'h w c -> c h w').unsqueeze(0)
            warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
            mask = (warped == 0).all(dim=1, keepdim=True).to(image.dtype)
            
            overlap = 1 - mask.sum().item() / (h*w)
            seq_info.views[query_idx].set_overlap_with_reference(idx, overlap)

            # visualize
            warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
            warped = ((warped + 1) * 127.5).astype(np.uint8)
            mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
            cv2.imwrite(os.path.join(seq_info.project_dir, f"warp_stage{seq_info.cur_stage}", f"{query_idx}_{idx}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(seq_info.project_dir, f"mask_stage{seq_info.cur_stage}", f"{query_idx}_{idx}.png"), (1-mask)*255)

        ref_cnt = len(ref_images)

        ref_images_stack = torch.cat(ref_images, dim=2)     # (1, 3, H * B, W)

        for i, flow in enumerate(ref_flows):
            flow[0,1] -= i * h
        ref_flows_stack = torch.cat(ref_flows, dim=2)       # (1, 2, H * B, W)

        ref_depths_stack = torch.cat(ref_depths, dim=0)     # (H * W * B,)
        ref_importance = 0.5 / ref_depths_stack
        ref_importance -= ref_importance.min()
        ref_importance /= ref_importance.max() + 1e-6
        ref_importance = ref_importance * 10 - 10

        ref_importance_chunks = []
        for i in range(ref_cnt):
            chunk = ref_importance[i*h*w:(i+1)*h*w]
            chunk = chunk.reshape(h, w, 1)
            chunk = rearrange(chunk, 'h w c -> c h w').unsqueeze(0)
            ref_importance_chunks.append(chunk)
        ref_importance = torch.cat(ref_importance_chunks, dim=2)  # (1, 1, H * B, W)

        warped = splatting_function('softmax', ref_images_stack, ref_flows_stack, ref_importance, eps=1e-6)
        warped = warped[:,:,:h,:]  # (1, 3, H, W)
        mask = (warped == 0).all(dim=1, keepdim=True).to(image.dtype)
        mask = mask[:,:,:h,:]  # (1, 1, H, W)

        if seq_info.reference[query_idx] == -1:
            known_area_ratio[query_idx] = 1 - mask.sum().item() / (h*w)
        seq_info.views[query_idx].set_warped_and_mask(warped, mask)

        # visualize
        warped = rearrange(warped[0], 'c h w -> h w c').detach().cpu().numpy()
        warped = ((warped + 1) * 127.5).astype(np.uint8)
        mask = rearrange(mask[0], 'c h w -> h w c').detach().cpu().numpy()
        cv2.imwrite(os.path.join(seq_info.project_dir, f"warp_stage{seq_info.cur_stage}", f"stacked_{query_idx}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(seq_info.project_dir, f"mask_stage{seq_info.cur_stage}", f"stacked_{query_idx}.png"), (1-mask)*255)

    plt.figure()
    x = np.arange(len(known_area_ratio))
    plt.plot(x, known_area_ratio, "k.", label="Known Area Ratio")
    plt.savefig(os.path.join(seq_info.project_dir, f"warp_stage{seq_info.cur_stage}", "known_area_ratio.png"))

    return known_area_ratio