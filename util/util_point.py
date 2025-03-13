import numpy as np
import open3d as o3d
from PIL import Image

def estimate_normals_towards_camera(points, camera_pos=(0,0,0), radius=0.1, max_nn=30):
    """
    points: (N,3) numpy array (float)
    camera_pos: (x, y, z) 형태의 카메라 위치
    radius, max_nn: estimate_normals에 사용할 KDTreeSearchParamHybrid 매개변수
    return: normals: (N,3) numpy array, 카메라 방향으로 법선이 향하도록 정렬됨
    """
    # Open3D PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 법선 추정
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn
    ))

    # 카메라 위치쪽으로 노멀 방향 정렬 (법선 플립)
    pcd.orient_normals_towards_camera_location(np.array(camera_pos))

    # 결과 반환
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals

def normal_to_rgb_image(normals, H, W, filename="normal_map.png"):
    """
    normals: (N, 3) or (H*W, 3) 형태, 성분 범위는 -1 ~ 1 사이라고 가정
    H, W: 이미지 크기
    filename: 저장할 파일 이름
    """
    # (H*W, 3) → (H, W, 3) reshape
    normals_2d = normals.reshape(H, W, 3)

    # [-1,1] 범위를 [0,1] → [0,255] 변환
    normals_clipped = np.clip(normals_2d, -1.0, 1.0)  # 혹시 범위 넘어간 경우
    normals_mapped = (normals_clipped * 0.5 + 0.5) * 255.0

    # uint8 변환
    normals_mapped = normals_mapped.astype(np.uint8)

    # PIL Image로 저장
    img = Image.fromarray(normals_mapped, mode='RGB')
    img.save(filename)
    print(f"Normal map saved as {filename}")