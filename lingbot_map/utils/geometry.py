# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation
try:
    from lietorch import SE3, Sim3
except ImportError:
    SE3 = Sim3 = None
import torch.nn.functional as F

try:
    from lingbot_map.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion
except ImportError:
    apply_distortion = iterative_undistortion = single_undistortion = None


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    【主函数】将深度图批量反投影到3D世界坐标系

    功能概述:
        将多帧深度图从2D图像空间转换到3D世界坐标空间。
        这是3D重建的核心步骤：深度图 → 相机坐标 → 世界坐标

    工作流程:
        1. 检查并转换输入数据格式（Tensor → NumPy）
        2. 对每一帧调用 depth_to_world_coords_points 进行反投影
        3. 将所有帧的结果堆叠成批处理输出

    坐标变换链:
        深度图 (H, W) → 相机坐标 (H, W, 3) → 世界坐标 (H, W, 3)

        其中变换公式为:
        P_camera = K_inv × [u, v, 1] × depth    (像素坐标 → 相机坐标)
        P_world = c2w × P_camera                 (相机坐标 → 世界坐标)

    Args:
        depth_map (np.ndarray): 深度图批次
            - 形状: (S, H, W, 1) 或 (S, H, W)
            - S: 序列长度（帧数）
            - H, W: 图像高度和宽度
            - 每个值表示该像素点到相机光心的距离

        extrinsics_cam (np.ndarray): 相机外参矩阵批次
            - 形状: (S, 3, 4)
            - 表示 camera-to-world (c2w) 变换矩阵
            - 包含相机的旋转R(3x3)和平移t(3x1)

        intrinsics_cam (np.ndarray): 相机内参矩阵批次
            - 形状: (S, 3, 3)
            - 标准内参格式: [fx,  0, cx]
                           [ 0, fy, cy]
                           [ 0,  0,  1]
            - fx, fy: 焦距（像素单位）
            - cx, cy: 光心坐标（像素单位）

    Returns:
        np.ndarray: 3D世界坐标点云
            - 形状: (S, H, W, 3)
            - 每个像素点对应一个3D世界坐标 (x, y, z)
    """
    # ========================================
    # 步骤1: 数据格式转换
    # ========================================
    # 如果输入是PyTorch Tensor，转换到NumPy数组
    # 这是因为在纯NumPy环境下处理更高效，且便于后续文件保存
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    # ========================================
    # 步骤2: 逐帧反投影处理
    # ========================================
    # 初始化结果列表，用于存储每帧的世界坐标点云
    world_points_list = []

    # 遍历每一帧（S帧）
    # 对每帧独立处理，调用单帧反投影函数
    for frame_idx in range(depth_map.shape[0]):
        # 调用单帧处理函数
        # squeeze(-1): 将 (H, W, 1) 压缩为 (H, W)
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),      # 当前帧深度图 (H, W)
            extrinsics_cam[frame_idx],              # 当前帧外参 (3, 4)
            intrinsics_cam[frame_idx]               # 当前帧内参 (3, 3)
        )
        world_points_list.append(cur_world_points)

    # ========================================
    # 步骤3: 结果堆叠
    # ========================================
    # 将所有帧的世界坐标沿新轴堆叠
    # 结果形状: (S, H, W, 3)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array 


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    【单帧处理函数】将单帧深度图转换到世界坐标系

    功能概述:
        这是核心反投影函数，实现深度图到世界坐标的完整变换链。
        处理流程: 深度图 → 相机坐标 → 世界坐标

    变换原理:
        ┌─────────────────────────────────────────────────────────────────┐
        │  1. 像素坐标 → 相机坐标 (通过内参矩阵逆变换)                    │
        │     x_cam = (u - cx) × depth / fx                              │
        │     y_cam = (v - cy) × depth / fy                              │
        │     z_cam = depth                                              │
        │                                                               │
        │     其中:                                                      │
        │     - u, v: 像素坐标                                          │
        │     - cx, cy: 光心坐标（内参）                                 │
        │     - fx, fy: 焦距（内参）                                     │
        │     - depth: 该像素的深度值                                    │
        │                                                               │
        │  2. 相机坐标 → 世界坐标 (通过外参矩阵变换)                      │
        │     P_world = R × P_camera + t                                │
        │     或写成: P_world = c2w × P_camera                          │
        │                                                               │
        │     其中:                                                      │
        │     - R: 相机朝向（旋转矩阵，3x3）                             │
        │     - t: 相机位置（平移向量，3x1）                             │
        │     - c2w: camera-to-world 变换矩阵                           │
        └─────────────────────────────────────────────────────────────────┘

    坐标系约定:
        - 相机坐标系 (OpenCV convention):
            X轴: 向右
            Y轴: 向下
            Z轴: 向前（拍摄方向）
            光心位于原点

        - 世界坐标系:
            由第一帧相机定义，或全局定义
            具体取决于extrinsic矩阵的参考系

    Args:
        depth_map (np.ndarray): 单帧深度图
            - 形状: (H, W)
            - 每个值表示像素点到相机光心的Z方向距离

        intrinsic (np.ndarray): 相机内参矩阵
            - 形状: (3, 3)
            - 格式: [fx,  0, cx]
                   [ 0, fy, cy]
                   [ 0,  0,  1]

        extrinsic (np.ndarray): 相机外参矩阵
            - 形状: (3, 4)
            - OpenCV相机坐标系约定
            - 表示 camera-from-world (w2c)，但实际代码中
              需要求逆得到 world-from-camera (c2w)
            - 格式: [R | t] (旋转矩阵R + 平移向量t)
                   [r11 r12 r13 | t1]
                   [r21 r22 r23 | t2]
                   [r31 r32 r33 | t3]

        eps (float): 有效深度阈值，用于过滤无效深度值

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 返回三个结果
            - world_coords_points: 世界坐标点云 (H, W, 3)
            - cam_coords_points: 相机坐标点云 (H, W, 3)
            - point_mask: 有效深度掩码 (H, W)，depth > eps的位置为True
    """
    # ========================================
    # 边界情况处理
    # ========================================
    if depth_map is None:
        return None, None, None

    # ========================================
    # 步骤1: 计算有效深度掩码
    # ========================================
    # 有效深度点: 深度值大于阈值eps（过滤0值和无效值）
    # 这对于后续过滤无效点云很重要
    point_mask = depth_map > eps

    # ========================================
    # 步骤2: 深度图 → 相机坐标
    # ========================================
    # 使用内参矩阵将深度图反投影到相机坐标系
    # 这是针孔相机模型的逆过程
    # 详细实现见 depth_to_cam_coords_points 函数
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # ========================================
    # 步骤3: 相机坐标 → 世界坐标
    # ========================================
    # 外参矩阵求逆: w2c → c2w
    # extrinsic 是 world-to-camera (w2c)，需要求逆得到 camera-to-world (c2w)
    #
    # 数学推导:
    #   w2c: P_camera = w2c × P_world
    #   c2w: P_world = c2w × P_camera = inv(w2c) × P_camera
    #
    # 注意: extrinsic[None] 添加batch维度，因为closed_form_inverse_se3需要batch输入
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    # 从4x4矩阵中提取旋转部分R(3x3)和平移部分t(3x1)
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]  # 旋转矩阵
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]   # 平移向量

    # ========================================
    # 步骤4: 应用变换
    # ========================================
    # 对每个相机坐标点应用 c2w 变换
    # 公式: P_world = R × P_camera + t
    # 或写成矩阵形式: P_world = c2w × P_camera (齐次坐标)
    #
    # 使用矩阵乘法实现批量变换:
    #   cam_coords_points: (H, W, 3)
    #   R_cam_to_world: (3, 3)
    #   结果: np.dot(cam_coords_points, R.T) + t → (H, W, 3)
    #
    # 注意: 这里用的是 R.T，因为:
    #   (H,W,3) × (3,3) 的矩阵乘法需要点作为行向量
    #   即 P_new = P_old × R.T，等价于 P_new = R × P_old (点作为列向量)
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask 


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    【像素→相机坐标转换函数】将深度图反投影到相机坐标系

    功能概述:
        这是针孔相机模型的逆过程，将2D像素坐标还原为3D相机坐标。
        在相机坐标系中，Z轴指向拍摄方向（前方），光心位于原点。

    数学原理:
        ┌─────────────────────────────────────────────────────────────────┐
        │  针孔相机模型的投影公式（正向）:                                │
        │     u = fx × (x_cam / z_cam) + cx                              │
        │     v = fy × (y_cam / z_cam) + cy                              │
        │                                                               │
        │  其中:                                                          │
        │     - (u, v): 像素坐标                                          │
        │     - (x_cam, y_cam, z_cam): 相机坐标                          │
        │     - fx, fy: 焦距（像素单位）                                  │
        │     - cx, cy: 光心坐标（像素单位）                              │
        │                                                               │
        │  反投影公式（逆向）:                                            │
        │     z_cam = depth_map[u, v]                                    │
        │     x_cam = (u - cx) × z_cam / fx                              │
        │     y_cam = (v - cy) × z_cam / fy                              │
        │                                                               │
        │  这个公式本质上是:                                             │
        │     [x_cam, y_cam, z_cam] = K_inv × [u, v, 1] × depth          │
        └─────────────────────────────────────────────────────────────────┘

    坐标系约定 (OpenCV convention):
        相机坐标系定义:
            - X轴: 向右（与图像u方向对应）
            - Y轴: 向下（与图像v方向对应）
            - Z轴: 向前（拍摄方向/深度方向）
            - 光心位于原点 (0, 0, 0)

        这意味着:
            - 图像中水平方向为相机X轴
            - 图像中垂直方向为相机Y轴
            - 深度值即为相机Z坐标

    Args:
        depth_map (np.ndarray): 单帧深度图
            - 形状: (H, W)
            - H: 图像高度（像素行数）
            - W: 图像宽度（像素列数）
            - 每个值表示像素点到相机光心的Z方向距离

        intrinsic (np.ndarray): 相机内参矩阵
            - 形状: (3, 3)
            - 标准格式（无畸变）:
                [fx,  0, cx]
                [ 0, fy, cy]
                [ 0,  0,  1]
            - fx, fy: 焦距（像素单位），fx≈fy对于大多数相机
            - cx, cy: 光心坐标（像素单位），通常cx≈W/2, cy≈H/2

    Returns:
        np.ndarray: 相机坐标系下的3D点云
            - 形状: (H, W, 3)
            - 每个像素对应一个3D点 (x_cam, y_cam, z_cam)
            - 无效深度点（depth=0）坐标为 (0, 0, 0)

    工作流程:
        1. 提取内参参数（fx, fy, cx, cy）
        2. 生成像素坐标网格 (u, v)
        3. 对每个像素应用反投影公式
        4. 堆叠成 (H, W, 3) 的3D点云数组
    """
    # ========================================
    # 步骤1: 获取图像尺寸并验证内参矩阵
    # ========================================
    H, W = depth_map.shape  # H: 高度（行数），W: 宽度（列数）

    # 验证内参矩阵形状必须为 3x3
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"

    # 验证内参矩阵无畸变（skew为0）
    # skew表示像素的非正方形畸变，现代相机通常为0
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # ========================================
    # 步骤2: 提取内参参数
    # ========================================
    # 从内参矩阵中提取焦距和光心坐标
    # fu (fx): u方向焦距，控制水平方向的缩放
    # fv (fy): v方向焦距，控制垂直方向的缩放
    # cu (cx): 光心的u坐标（水平方向），通常是图像宽度的中心
    # cv (cy): 光心的v坐标（垂直方向），通常是图像高度的中心
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]  # 焦距
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]  # 光心坐标

    # ========================================
    # 步骤3: 生成像素坐标网格
    # ========================================
    # np.meshgrid 生成像素坐标网格
    # np.arange(W): [0, 1, 2, ..., W-1]（列索引）
    # np.arange(H): [0, 1, 2, ..., H-1]（行索引）
    #
    # meshgrid 参数顺序: 第一个参数对应列（u），第二个对应行（v）
    # 结果:
    #   u: (H, W) 矩阵，每行为 [0, 1, ..., W-1]，表示每个像素的列索引
    #   v: (H, W) 矩阵，每列为 [0, 1, ..., H-1]，表示每个像素的行索引
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # ========================================
    # 步骤4: 应用反投影公式
    # ========================================
    # 核心反投影公式:
    #
    # 对于像素 (u, v)，其深度为 depth_map[v, u]
    # （注意: 数组索引是 [行, 列] 即 [v, u]）
    #
    # 反投影计算:
    #   x_cam = (u - cx) × depth / fx
    #   y_cam = (v - cy) × depth / fy
    #   z_cam = depth
    #
    # 数学解释:
    #   - (u - cx) / fx: 将像素u坐标转换为归一化相机坐标（单位：1/深度）
    #   - 乘以depth后得到实际相机x坐标（单位：米）
    #   - 同理适用于y方向
    #   - z坐标直接等于深度值
    #
    # 物理意义:
    #   - 深度值 depth 代表像素点到光心的Z方向距离
    #   - 相机坐标系中，Z轴指向拍摄方向
    #   - 因此 z_cam = depth 是合理的

    # 计算 x_cam: 水平方向（图像u方向 → 相机X轴）
    # (u - cu): 像素相对于光心的水平偏移
    # 除以fu后乘以depth: 将偏移缩放到实际物理尺度
    x_cam = (u - cu) * depth_map / fu

    # 计算 y_cam: 垂直方向（图像v方向 → 相机Y轴）
    # (v - cv): 像素相对于光心的垂直偏移
    # 除以fv后乘以depth: 将偏移缩放到实际物理尺度
    y_cam = (v - cv) * depth_map / fv

    # 计算 z_cam: 深度方向（相机Z轴）
    # 直接等于深度值，因为相机坐标系Z轴定义为深度方向
    z_cam = depth_map

    # ========================================
    # 步骤5: 堆叠成3D点云
    # ========================================
    # np.stack((x_cam, y_cam, z_cam), axis=-1)
    # 将三个 (H, W) 矩阵沿最后一个维度堆叠
    # 结果: (H, W, 3)，每个元素为 (x_cam, y_cam, z_cam)
    #
    # astype(np.float32): 转换为单精度浮点数
    # - 减少内存占用（相比float64）
    # - 对于大多数应用精度足够
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    【SE3矩阵求逆函数】计算SE3变换矩阵的逆矩阵（闭式解）

    功能概述:
        对一批SE3（特殊欧氏群）变换矩阵求逆。
        SE3表示3D空间中的刚性变换（旋转+平移）。

    数学原理:
        ┌─────────────────────────────────────────────────────────────────┐
        │  SE3矩阵定义:                                                   │
        │     SE3 = [R | t]  (4x4或3x4矩阵)                               │
        │           [0 | 1]                                              │
        │                                                               │
        │     其中:                                                      │
        │     - R: 3x3旋转矩阵（正交矩阵，R^T × R = I）                   │
        │     - t: 3x1平移向量                                           │
        │     - 最后一行 [0 | 1] 使矩阵成为齐次变换                       │
        │                                                               │
        │  SE3逆矩阵的闭式解:                                            │
        │     inv(SE3) = [R^T   | -R^T × t]                              │
        │                [0    |    1    ]                               │
        │                                                               │
        │  证明:                                                         │
        │     SE3 × inv(SE3) = I                                        │
        │     [R | t] × [R^T | -R^T×t]                                  │
        │     = [R×R^T | R×(-R^T×t) + t]                                │
        │     = [I | -t + t]                                            │
        │     = [I | 0]                                                 │
        │     = I                                                       │
        │                                                               │
        │  物理意义:                                                     │
        │     如果 SE3 表示 "A坐标系 → B坐标系" 的变换                   │
        │     则 inv(SE3) 表示 "B坐标系 → A坐标系" 的变换                │
        │                                                               │
        │     例如:                                                      │
        │     - w2c (world-to-camera): 世界坐标 → 相机坐标              │
        │     - inv(w2c) = c2w: 相机坐标 → 世界坐标                      │
        └─────────────────────────────────────────────────────────────────┘

    为什么需要这个函数:
        在3D重建中，模型输出的extrinsic矩阵通常是 w2c（world-to-camera）。
        要将相机坐标系下的点转换到世界坐标系，需要 c2w = inv(w2c)。
        这个函数提供了高效、精确的SE3逆矩阵计算。

    Args:
        se3: SE3变换矩阵批次
            - 类型: np.ndarray 或 torch.Tensor
            - 形状: (N, 4, 4) 或 (N, 3, 4)
            - N: 批次大小（矩阵数量）
            - 4x4格式: [R | t]
                        [0 | 1]
            - 3x4格式: 只包含 [R | t] 部分，最后一行 [0 | 1] 隐含

        R (optional): 预提取的旋转矩阵
            - 形状: (N, 3, 3)
            - 如果提供，可避免从se3中提取，提高效率

        T (optional): 预提取的平移向量
            - 形状: (N, 3, 1)
            - 如果提供，可避免从se3中提取，提高效率

    Returns:
        与输入类型相同的逆SE3矩阵
            - 形状: (N, 4, 4)
            - 类型: np.ndarray（如果输入是numpy）或 torch.Tensor（如果输入是tensor）
            - 设备: 与输入相同（GPU/CPU）

    工作流程:
        1. 判断输入类型（numpy或torch）
        2. 验证矩阵形状
        3. 从se3中提取R和T（如果未提供）
        4. 计算 R^T（旋转矩阵的逆）
        5. 计算 -R^T × t（逆矩阵的平移部分）
        6. 构造完整的逆矩阵
    """
    # ========================================
    # 步骤1: 判断输入类型
    # ========================================
    # 检查输入是NumPy数组还是PyTorch张量
    # 两种类型的矩阵操作语法略有不同，需要分支处理
    is_numpy = isinstance(se3, np.ndarray)

    # ========================================
    # 步骤2: 验证矩阵形状
    # ========================================
    # SE3矩阵必须是 (N, 4, 4) 或 (N, 3, 4)
    # - 4x4: 完整的齐次变换矩阵
    # - 3x4: 简化格式，最后一行 [0 | 1] 隐含
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # ========================================
    # 步骤3: 提取R和T（如果未提供）
    # ========================================
    # R: 旋转矩阵部分，位于左上角 3x3
    # T: 平移向量部分，位于最后一列的前3个元素
    #
    # 如果调用者已经预提取了R和T，直接使用可提高效率
    # 这在某些场景下有用，如需要多次调用此函数时
    if R is None:
        R = se3[:, :3, :3]  # (N, 3, 3) 旋转矩阵
    if T is None:
        T = se3[:, :3, 3:]  # (N, 3, 1) 平移向量，保持列向量形式

    # ========================================
    # 步骤4: 根据类型分支处理
    # ========================================
    if is_numpy:
        # ========================================
        # NumPy分支
        # ========================================
        # 计算 R^T（旋转矩阵的转置）
        # 对于正交矩阵（旋转矩阵），R^T = R^{-1}（逆矩阵）
        # np.transpose(R, (0, 2, 1)): 对每个批次矩阵做转置
        #   - (0, 2, 1) 表示: 保持第0维（batch），交换第1和第2维
        R_transposed = np.transpose(R, (0, 2, 1))

        # 计算 -R^T × t（逆矩阵的平移部分）
        # np.matmul: 批次矩阵乘法
        # R_transposed: (N, 3, 3)
        # T: (N, 3, 1)
        # 结果: (N, 3, 1)
        #
        # 数学推导:
        #   原变换: P_new = R × P_old + t
        #   逆变换: P_old = R^T × P_new - R^T × t
        #   因此逆矩阵的平移为 -R^T × t
        top_right = -np.matmul(R_transposed, T)

        # 构造结果矩阵模板：4x4单位矩阵，复制N份
        # np.tile(np.eye(4), (len(R), 1, 1)): 创建N个4x4单位矩阵
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))

    else:
        # ========================================
        # PyTorch分支
        # ========================================
        # 计算 R^T（旋转矩阵的转置）
        # torch.Tensor.transpose(1, 2): 交换第1和第2维
        R_transposed = R.transpose(1, 2)  # (N, 3, 3)

        # 计算 -R^T × t（逆矩阵的平移部分）
        # torch.bmm: 批次矩阵乘法（batch matrix multiplication）
        top_right = -torch.bmm(R_transposed, T)  # (N, 3, 1)

        # 构造结果矩阵模板：4x4单位矩阵，复制N份
        # torch.eye(4, 4)[None].repeat(len(R), 1, 1): 创建N个4x4单位矩阵
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)

        # 确保数据类型和设备与输入一致
        # 这对于GPU计算很重要，避免设备不匹配错误
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    # ========================================
    # 步骤5: 填充逆矩阵
    # ========================================
    # 逆矩阵结构:
    #   inv(SE3) = [R^T   | -R^T × t]
    #              [0     |    1    ]
    #
    # inverted_matrix[:, :3, :3]: 前三行前三列（旋转部分）
    # inverted_matrix[:, :3, 3:]: 前三行最后一列（平移部分）
    inverted_matrix[:, :3, :3] = R_transposed  # 填充 R^T
    inverted_matrix[:, :3, 3:] = top_right      # 填充 -R^T × t

    # 第四行 [0 | 1] 已由单位矩阵模板设置好，无需修改

    return inverted_matrix

def closed_form_inverse_se3_general(se3, R=None, T=None):
    """
    支持任意 batch 维度的 SE3 逆运算
    se3: (..., 4, 4) 或 (..., 3, 4)
    """
    batch_shape = se3.shape[:-2]
    if R is None:
        R = se3[..., :3, :3]
    if T is None:
        T = se3[..., :3, 3:]
    R_transposed = R.transpose(-2, -1)
    top_right = -R_transposed @ T
    # 构造单位阵
    eye = torch.eye(4, 4, dtype=R.dtype, device=R.device)
    inverted_matrix = eye.expand(*batch_shape, 4, 4).clone()
    inverted_matrix[..., :3, :3] = R_transposed
    inverted_matrix[..., :3, 3:] = top_right
    return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Apply distortion if distortion_params are provided
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """

    # We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
    # otherwise we can use something like
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # Apply iterative undistortion
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized

## Droid SLAM Part

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def projective_transform(
    poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False
):
    """map points from ii->jj"""

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    # Gij.data[:, ii == jj] = torch.as_tensor(
    #     [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda"
    # )
    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)

        Jz = Gij[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """optical flow induced by camera motion"""

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht, device=disps.device, dtype=torch.float),
        torch.arange(wd, device=disps.device, dtype=torch.float),
        indexing="ij",
    )
    
    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[..., :2] - coords0, valid

def all_pairs_distance_matrix(poses, beta=2.5):
    """ compute distance matrix between all pairs of poses """
    poses = np.array(poses, dtype=np.float32)
    poses[:,:3] *= beta # scale to balence rot + trans
    poses = SE3(torch.from_numpy(poses))

    r = (poses[:,None].inv() * poses[None,:]).log()
    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[..., :3, :3]).as_quat()
    return np.concatenate([pose[..., :3, 3], q], axis=-1)

def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1, val1 = induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2, val2 = induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix


def compute_distance_matrix_flow2(poses, disps, intrinsics, beta=0.4):
    """ compute flow magnitude between all pairs of frames """
    # if not isinstance(poses, SE3):
    #     poses = torch.from_numpy(poses).float().cuda()[None]
    #     poses = SE3(poses).inv()

    #     disps = torch.from_numpy(disps).float().cuda()[None]
    #     intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    MAX_FLOW = 128.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1a, val1a = induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s], tonly=True)
        flow1b, val1b = induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2a, val2a = induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s], tonly=True)
        flow2b, val2b = induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])

        flow1 = flow1a + beta * flow1b
        val1 = val1a * val2b

        flow2 = flow2a + beta * flow2b
        val2 = val2a * val2b
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.8] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht, dtype=torch.float, **kwargs),
        torch.arange(wd, dtype=torch.float, **kwargs),
        indexing="ij",
    )

    return torch.stack([x, y], dim=-1)


def iproj(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht, device=disps.device, dtype=torch.float),
        torch.arange(wd, device=disps.device, dtype=torch.float),
        indexing="ij",
    )

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[..., -1] = 1.0
        return pts, J

    return pts, None


def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """pinhole camera projection"""
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D * d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack(
            [
                fx * d,
                o,
                -fx * X * d * d,
                o,
                o,
                fy * d,
                -fy * Y * d * d,
                o,
                # o,     o,    -D*d*d,  d,
            ],
            dim=-1,
        ).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def actp(Gij, X0, jacobian=False):
    """action on point cloud"""
    X1 = Gij[:, :, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t
