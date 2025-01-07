import os

import cv2
import numpy as np
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

def compute_optical_flow(P, R_source, T_source, R_query, T_query, K):
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
    pts_source, _ = project_points(P, R_source, T_source, K)  # (N, 2)
    
    # 쿼리 뷰에서의 2D 투영
    pts_query, depth_query = project_points(P, R_query, T_query, K)  # (N, 2)
    
    # for i in range(10):
    #     x = round(pts_source[i][0],2)
    #     y = round(pts_source[i][1],2)
    #     target_x = i
    #     target_y = 0
    #     if abs(x - target_x) > 5 or abs(y - target_y) > 5:
    #         print(f"Projection warning! : {x}, {y} -> {target_x}, {target_y}")

    # idx = np.random.choice(len(pts_source), 10, replace=False)
    # for i in idx:
    #     print(round(pts_source[i][0],2), round(pts_source[i][1],2), "->", round(pts_query[i][0],2), round(pts_query[i][1],2))

    # Optical Flow 계산
    flow = pts_query - pts_source  # (N, 2)
    
    return flow, depth_query

def warp_reference(images, extrinsics, intrinsics, trajectory, reference, i, save_dir):
    os.makedirs(os.path.join(save_dir, "warp"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "mask"), exist_ok=True)

    for query_idx in range(reference):
        if reference[query_idx] != -1:
            continue
    
        query_R = trajectory[query_idx][:3, :3]
        query_T = trajectory[query_idx][:3, 3]

        for idx in range(reference):
            if reference[idx] == -1:
                continue
            
            ref_idx = reference[idx]

            K = np.array([
                [intrinsics["focals"][ref_idx], 0, intrinsics["principal_points"][ref_idx][0]],
                [0, intrinsics["focals"][ref_idx], intrinsics["principal_points"][ref_idx][1]],
                [0, 0, 1]
            ])

            ref_image = images[ref_idx]
            h,w = ref_image.shape[:2]
            
            ref_pts3d = np.load(os.path.join(save_dir, "recon", f"pts3d_{ref_idx}.npy"))
            ref_pts3d = ref_pts3d.reshape(-1, 3)

            reference_R = extrinsics[ref_idx][:3, :3]
            reference_T = extrinsics[ref_idx][:3, 3]
            assert np.allclose(query_R, reference_R), f"Rotation does not match at index {idx}: {query_R}, {reference_R}"
            assert np.allclose(query_T, reference_T), f"Translation does not match at index {idx}: {query_T}, {reference_T}"

            flow, depth = compute_optical_flow(ref_pts3d, reference_R, reference_T, query_R, query_T, K)
            flow = flow.reshape(h, w, 2)

            importance = 0.5 / depth
            importance -= importance.amin((0, 1), keepdims=True)
            importance /= importance.max((0, 1), keepdims=True) + 1e-6
            importance = importance * 10 - 10
            importance = importance.reshape(h, w, 1)

            flow = rearrange(flow, 'h w c -> c h w')
            importance = rearrange(importance, 'h w c -> c h w')
            warped = splatting_function('softmax', ref_image, flow, importance, eps=1e-6)
            mask = (warped == 0).all(dim=0).float()

            cv2.imwrite(os.path.join(save_dir, "warp", f"warped_{query_idx}_{ref_idx}.png"), warped)
            cv2.imwrite(os.path.join(save_dir, "mask", f"mask_{query_idx}_{ref_idx}.png"), mask)