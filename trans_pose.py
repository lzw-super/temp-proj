import torch
import numpy as np 
'''
先将2d深度信息转换成3d相机坐标 
同时将w2c矩阵取逆转换成c2w矩阵
再利用c2w矩阵将相机坐标转换成世界坐标
最后计算对齐矩阵（包含对齐第一帧以及适配坐标系）并将世界坐标对齐到第一帧的坐标系下得到所需点云数据
'''
def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
    ) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array

def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask

def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def export_raw_ply(predictions, images, output_path, conf_threshold=1.5, downsample_factor=10, use_depth=True):
    """Export world_points to PLY format with proper coordinate alignment and filtering.

    Applies the same processing as viewer/GLB export:
    1. Coordinate alignment (first camera frame, OpenGL convention, 180 deg rotation)
    2. Confidence threshold filtering (if conf_threshold > 0)
    3. Downsample (if downsample_factor > 1)

    Args:
        predictions: Dictionary containing 'world_points', 'extrinsic', 'depth', 'depth_conf', 'intrinsic'
        images: Image tensor (S, 3, H, W) or (B, S, 3, H, W) - may need squeeze
        output_path: Path to save the PLY file
        conf_threshold: Confidence threshold (higher = fewer, higher-quality points). 0 = no filter.
        downsample_factor: Downsample factor (1 = all points, 10 = every 10th point)
        use_depth: If True, use depth-based points (same as viewer default).
                   If False, use model's world_points directly.
    """
    from scipy.spatial.transform import Rotation
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    extrinsics = predictions.get("extrinsic")
    intrinsics = predictions.get("intrinsic")
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")
    world_points_direct = predictions.get("world_points")
    world_points_conf = predictions.get("world_points_conf")

    if extrinsics is None or intrinsics is None:
        print("Error: 'extrinsic' or 'intrinsic' not found in predictions")
        return

    # Handle both tensor and numpy array
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()
    if isinstance(depth_conf, torch.Tensor):
        depth_conf = depth_conf.numpy()
    if world_points_direct is not None and isinstance(world_points_direct, torch.Tensor):
        world_points_direct = world_points_direct.numpy()
    if world_points_conf is not None and isinstance(world_points_conf, torch.Tensor):
        world_points_conf = world_points_conf.numpy()

    # Choose point cloud source (same logic as viewer)
    if use_depth and depth is not None:
        print("Using depth-based points (same as viewer default)")
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)  #应用内外参转换成世界坐标  
        conf = depth_conf
    else:
        print("Using model's world_points directly")
        if world_points_direct is None:
            print("Error: 'world_points' not found in predictions")
            return
        world_points = world_points_direct
        conf = world_points_conf if world_points_conf is not None else depth_conf

    # Handle images - may need squeeze for batch dimension
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Debug: print shapes before processing
    print(f"Before processing:")
    print(f"  world_points shape: {world_points.shape}")
    print(f"  images shape: {images.shape}")
    print(f"  images dtype: {images.dtype}")
    print(f"  images value range: min={images.min():.4f}, max={images.max():.4f}")

    # Squeeze batch dimension if present (B, S, 3, H, W) -> (S, 3, H, W)
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
        print(f"  Squeezed batch dim, images shape now: {images.shape}")

    # Convert (S, 3, H, W) -> (S, H, W, 3) for color extraction
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
        print(f"  Transposed to (S, H, W, 3), images shape now: {images.shape}")

    # Final shapes
    print(f"After processing:")
    print(f"  world_points shape: {world_points.shape}")
    print(f"  images shape: {images.shape}")

    # Verify dimensions match
    S_pts = world_points.shape[0]
    S_imgs = images.shape[0]
    if S_pts != S_imgs:
        print(f"Warning: dimension mismatch - world_points has {S_pts} frames, images has {S_imgs}")
        # Try to match
        if S_imgs > S_pts:
            images = images[:S_pts]
        elif S_pts > S_imgs:
            # Can't fix this easily
            print("Error: insufficient images for all point frames")
            return

    # Compute coordinate alignment transformation (same as GLB export)
    extrinsic_0_4x4 = np.eye(4)
    extrinsic_0_4x4[:3, :4] = extrinsics[0]

    # OpenGL conversion matrix (flip Y and Z)
    opengl_conversion = np.eye(4)
    opengl_conversion[1, 1] = -1
    opengl_conversion[2, 2] = -1

    # Align rotation (180 deg around Y)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    initial_transform = np.linalg.inv(extrinsic_0_4x4) @ opengl_conversion @ align_rotation   # 初始变换矩阵（同时包含对应的旋转和平移适配）
    print("Applying coordinate alignment transformation...")

    # Collect all points across all frames
    all_points = []
    all_colors = []

    S = world_points.shape[0]
    total_raw_pts = 0
    total_filtered_pts = 0

    for i in range(S):
        pts = world_points[i].reshape(-1, 3)  # (H*W, 3)
        cols = images[i].reshape(-1, 3)  # (H*W, 3) after transpose
        conf_i = conf[i].reshape(-1) if conf is not None else np.ones(len(pts))

        total_raw_pts += len(pts)

        # Build valid mask: non-zero points + finite + confidence threshold
        valid_mask = np.isfinite(pts).all(axis=1) & np.any(pts != 0, axis=1)

        if conf_threshold > 0:
            valid_mask = valid_mask & (conf_i > conf_threshold)

        if np.sum(valid_mask) == 0:
            continue

        pts_valid = pts[valid_mask]
        cols_valid = cols[valid_mask]

        # Debug: print color info for first frame
        if i == 0:
            print(f"Frame 0 color check:")
            print(f"  cols_valid shape: {cols_valid.shape}")
            print(f"  cols_valid dtype: {cols_valid.dtype}")
            print(f"  cols_valid sample values: {cols_valid[:5]}")
            print(f"  cols_valid min: {cols_valid.min()}, max: {cols_valid.max()}")

        # Apply transformation
        pts_aligned = pts_valid @ initial_transform[:3, :3].T + initial_transform[:3, 3]

        # Convert colors to uint8 (from [0,1] float range)
        if cols_valid.dtype != np.uint8:
            # Check value range before conversion
            col_min, col_max = cols_valid.min(), cols_valid.max()
            if col_min < 0 or col_max > 1:
                print(f"Warning: color values out of [0,1] range: min={col_min:.3f}, max={col_max:.3f}")
                # Try to normalize if values look like 0-255 range
                if col_max > 1 and col_max <= 255:
                    cols_valid = cols_valid / 255.0
            cols_valid = (np.clip(cols_valid, 0, 1) * 255).astype(np.uint8)
        all_points.append(pts_aligned)
        all_colors.append(cols_valid)
        total_filtered_pts += len(pts_aligned)

    if not all_points:
        print("Error: no valid points to export")
        return

    vertices = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Debug: check final colors before writing
    print(f"Final colors check:")
    print(f"  colors shape: {colors.shape}")
    print(f"  colors dtype: {colors.dtype}")
    print(f"  colors sample values: {colors[:5]}")
    print(f"  colors min: {colors.min()}, max: {colors.max()}")

    # Downsample
    if downsample_factor > 1:
        indices = np.arange(0, len(vertices), downsample_factor)
        vertices = vertices[indices]
        colors = colors[indices]
        print(f"Downsampled: {len(vertices)} points (factor={downsample_factor})")

    # Write binary PLY
    try:
        with open(output_path, 'wb') as f:
            # Header
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
            header += "property float x\nproperty float y\nproperty float z\n"
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            header += "end_header\n"
            f.write(header.encode('ascii'))

            # Binary vertex data
            vertex_data = np.zeros(len(vertices), dtype=np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', '<u1'), ('green', '<u1'), ('blue', '<u1')
            ]))
            vertex_data['x'] = vertices[:, 0]
            vertex_data['y'] = vertices[:, 1]
            vertex_data['z'] = vertices[:, 2]
            vertex_data['red'] = colors[:, 0]
            vertex_data['green'] = colors[:, 1]
            vertex_data['blue'] = colors[:, 2]
            vertex_data.tofile(f)

        print(f"PLY exported: {output_path}")
        print(f"  Total raw points: {total_raw_pts:,}")
        print(f"  After confidence filter: {total_filtered_pts:,}")
        print(f"  Final exported: {len(vertices):,}")

    except Exception as e:
        print(f"PLY export error: {e}")

