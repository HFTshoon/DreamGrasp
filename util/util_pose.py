import numpy as np
import open3d as o3d

def rotation_matrix_about_axis(axis, angle):
    """
    Returns a 3x3 rotation matrix that rotates by `angle` around the given `axis`.
    Uses Rodrigues' rotation formula.
    axis: 3D vector (assumed normalized).
    angle: scalar (radians).
    """
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c   ]
    ])
    return R

def angle_about_axis(z_axis, from_vec, to_vec):
    """
    Compute the angle needed to rotate `from_vec` into `to_vec` around the axis `z_axis`.
    Both from_vec and to_vec are assumed orthonormal to z_axis (i.e., all are perpendicular).
    The sign of the angle is determined by right-hand rule about z_axis.
    """    
    # Project cross product onto z_axis => magnitude ~ sin(angle), sign by direction
    cross_ft = np.cross(from_vec, to_vec)
    # Dot with z_axis for sign:
    sin_val = np.dot(cross_ft, z_axis)
    
    # Dot product => cos(angle)
    cos_val = np.dot(from_vec, to_vec)
    
    angle = np.arctan2(sin_val, cos_val)
    return angle

def generate_smooth_camera_path(R1, T1, R2, T2, N):
    """
    두 카메라의 회전 행렬과 위치 벡터가 주어졌을 때,
    카메라 레이 사이의 가장 가까운 점의 중점을 구의 중심으로 설정하고,
    구의 단면 원주를 따라 부드럽게 이동하는 카메라 경로를 생성합니다.

    Parameters:
    - R1, T1: 첫 번째 카메라의 회전 행렬 (3x3)과 위치 벡터 (3D 벡터)
    - R2, T2: 두 번째 카메라의 회전 행렬 (3x3)과 위치 벡터 (3D 벡터)
    - N: 스텝 수 (양의 정수)

    Returns:
    - positions: 카메라 위치 리스트 (N+1 개의 3D 벡터)
    - rotations: 회전 행렬 리스트 (N+1 개의 3x3 행렬)
    """
    print("T1: ", T1)
    print("R1: ")
    print(R1)
    print("T2: ", T2)
    print("R2: ")
    print(R2)
    # 카메라 레이 방향 벡터 (Z축)
    D1 = R1[:, 2]
    D2 = R2[:, 2]

    # 두 레이 사이의 가장 가까운 점 계산
    T1 = np.array(T1, dtype=float)
    T2 = np.array(T2, dtype=float)
    D1 = D1 / np.linalg.norm(D1)
    D2 = D2 / np.linalg.norm(D2)
    T21 = T2 - T1

    a = np.dot(D1, D1)
    b = np.dot(D1, D2)
    c = np.dot(D2, D2)
    d = np.dot(D1, T21)
    e = np.dot(D2, T21)
    print(a,b,c,d,e)

    denom = a * c - b * b
    if np.abs(denom) < 1e-6:
        print("Warning: Camera rays are parallel.")
        # 레이가 평행한 경우
        s = 0
        t = d / c
    else:
        s = (- b * e + c * d) / denom
        t = (- a * e + b * d) / denom

    P1_closest = T1 + s * D1
    P2_closest = T2 + t * D2

    # 구의 중심 (두 가장 가까운 점의 중점)
    C = (P1_closest + P2_closest) / 2
    print("s: ", s, "t: ", t)
    print("P1_closest: ", P1_closest)
    print("P2_closest: ", P2_closest)
    print("C: ", C)

    # 각 카메라 위치에서의 깊이 (구의 중심으로부터의 거리)
    depth1 = np.linalg.norm(T1 - C)
    depth2 = np.linalg.norm(T2 - C)

    # 카메라 위치를 반지름 1인 구에 투영
    T1_dir = T1 - C
    T2_dir = T2 - C
    T1_dir_norm = T1_dir / np.linalg.norm(T1_dir)
    T2_dir_norm = T2_dir / np.linalg.norm(T2_dir)
    T1_proj = C + T1_dir_norm
    T2_proj = C + T2_dir_norm

    # 두 투영된 위치 벡터 사이의 각도 계산
    dot = np.dot(T1_dir_norm, T2_dir_norm)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    print("omega: ", omega)
    print("sin_omega: ", sin_omega)

    up_vector = np.array([0, 1, 0])
    x1_pseudo = np.cross(D1, up_vector)
    x1_pseudo /= np.linalg.norm(x1_pseudo)
    print("x1_pseudo: ", x1_pseudo)
    pseudo_gap_1 = angle_about_axis(D1, x1_pseudo, R1[:, 0])
    x2_pseudo = np.cross(D2, up_vector)
    x2_pseudo /= np.linalg.norm(x2_pseudo)
    print("x2_pseudo: ", x2_pseudo)
    pseudo_gap_2 = angle_about_axis(D2, x2_pseudo, R2[:, 0])
    print("pseudo_gap_1: ", pseudo_gap_1)
    print("pseudo_gap_2: ", pseudo_gap_2)

    positions = []
    rotations = []

    positions.append(T1)
    rotations.append(R1)

    for i in range(1, N-1):
        t = i / (N-1)
        # 깊이를 큐빅 보간으로 부드럽게 변화
        t_cubic = 3 * t ** 2 - 2 * t ** 3
        depth = depth1 + (depth2 - depth1) * t_cubic

        # 구면 선형 보간 (SLERP)으로 방향 벡터 보간
        if sin_omega < 1e-6:
            dir_interp = T1_dir_norm
        else:
            a = np.sin((1 - t) * omega) / sin_omega
            b = np.sin(t * omega) / sin_omega
            dir_interp = a * T1_dir_norm + b * T2_dir_norm
            dir_interp /= np.linalg.norm(dir_interp)

        # 현재 위치 계산
        position = C + dir_interp * depth
        positions.append(position)

        watching_position = P1_closest * (1 - t) + P2_closest * t
        D_vec = watching_position - position
        D = D_vec / np.linalg.norm(D_vec)
        
        # 회전 행렬 계산 (카메라가 해당 방향을 바라보도록)
        z_axis = D

        x_pseudo = np.cross(z_axis, up_vector)
        x_pseudo /= np.linalg.norm(x_pseudo)
        y_axis = np.cross(z_axis, x_pseudo)
        
        pseudo_gap = pseudo_gap_1 * (1 - t) + pseudo_gap_2 * t
        pseudo_gap_matrix = rotation_matrix_about_axis(z_axis, pseudo_gap)

        x_axis = pseudo_gap_matrix @ x_pseudo
        x_axis /= np.linalg.norm(x_axis)
        y_axis = pseudo_gap_matrix @ y_axis
        y_axis /= np.linalg.norm(y_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))
        rotations.append(R)
    
    positions.append(T2)
    rotations.append(R2)

    # for i in range(len(positions)):
    #     print("Position: ", positions[i])
    #     print("Rotation: ")
    #     print(rotations[i])

    return positions, rotations

if __name__ == "__main__":
    # 사용 예시
    # R1 = np.eye(3)
    # T1 = [0, 0, 0]
    # R2 = np.array([
    #     [0.6, 0, -0.8],
    #     [0, 1, 0],
    #     [0.8, 0, 0.6]
    # ])
    # T2 = [1, 0, 0]
    # N = 10

    R1 = np.array([
        [0.82418, -0.035466, -0.56522],
        [0.053797, 0.99843, 0.015795],
        [0.56377, -0.043425, 0.82479]
    ])
    T1 = [0.16302, 0.0064381, 0.054947]
    # scale 10
    T1 = [x * 10 for x in T1]
    R2 = np.array([
        [0.99997, -0.0063155, 0.0053283],
        [0.0063151, 0.99998, 0.000092527],
        [-0.0053288, -0.000058875, 0.99999]
    ])

    T2 = [0, 0, 0]
    N = 10
    
    positions, rotations = generate_smooth_camera_path(R1, T1, R2, T2, N)

    # 결과 출력
    import matplotlib.pyplot as plt
    from util.extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
    visualizer = CameraPoseVisualizer([-3, 3], [-3, 3], [0, 3])
    h = 0
    for i in range(len(positions)):        
        R = np.array(rotations[i])
        T = np.array(positions[i]).reshape(-1,1)
        # R_i = R.T
        # T_i = -np.matmul(R.T, T)
        matrix = np.concatenate((np.concatenate((R, T), axis=1), np.array([[0,0,0,1]])), axis=0)
        visualizer.extrinsic2pyramid(matrix, plt.cm.rainbow(h / len(positions)), focal_len_scaled=1)
        h += 1
    visualizer.show()