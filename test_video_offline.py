"""Test script for save_pointcloud_video_offline function.

Usage:
    python test_video_offline.py --data_dir /path/to/exported_data --output_path output.mp4

    # Additional options:
    python test_video_offline.py --data_dir ./video_data --output_path test_result.mp4 \
        --fps 10 --resolution 1920x1080 --conf_threshold 1.5 --downsample_factor 10 --mode accumulate

The data_dir should contain:
    - extrinsic.npy: (S, 3, 4) camera extrinsics (c2w)
    - intrinsic.npy: (S, 3, 3) camera intrinsics
    - depth.npy: (S, H, W, 1) depth maps
    - depth_conf.npy: (S, H, W) depth confidence
    - images.npy: (S, H, W, 3) RGB images

Export data using demo.py:
    python demo.py --model_path /path/to/checkpoint.pt --image_folder /path/to/images \
        --export_video_data ./video_data --no_viewer
"""

import argparse
import glob
import logging
import os
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
# from demo import save_pointcloud_video_offline 
from lingbot_map.utils.geometry import closed_form_inverse_se3_general , closed_form_inverse_se3
def save_pointcloud_video_offline(predictions, images, output_path, fps=30, resolution="1920x1080",
                                   conf_threshold=1.5, downsample_factor=10, mode="accumulate"):
    """Save point cloud video using Open3D offline rendering (GPU accelerated, no browser needed).

    Args:
        predictions: Dictionary containing 'world_points', 'extrinsic', 'depth', etc.
        images: Image tensor for color
        output_path: Path to save the video
        fps: Frame rate
        resolution: Resolution string
        conf_threshold: Confidence threshold for filtering points
        downsample_factor: Downsample factor
        mode: "accumulate" (show all points up to current) or "single" (show only current frame)
    """
    import subprocess
    import tempfile
    import shutil
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    width, height = map(int, resolution.split('x'))

    # Get point cloud data
    extrinsics = predictions.get("extrinsic")
    intrinsics = predictions.get("intrinsic")
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")

    if extrinsics is None or intrinsics is None or depth is None:
        print("Error: Required data not found in predictions")
        return

    # Convert to numpy
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()
    if isinstance(depth_conf, torch.Tensor):
        depth_conf = depth_conf.numpy()
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    # Handle batch dimension
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)

    # Compute world points
    print("Computing world points from depth...")
    world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)

    # Compute coordinate alignment transformation (same as export_raw_ply)
    extrinsic_0_4x4 = np.eye(4)
    extrinsic_0_4x4[:3, :4] = extrinsics[0]

    # OpenGL conversion matrix (flip Y and Z)
    opengl_conversion = np.eye(4)
    opengl_conversion[1, 1] = -1
    opengl_conversion[2, 2] = -1

    # Align rotation (180 deg around Y)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    initial_transform = np.linalg.inv(extrinsic_0_4x4) @ opengl_conversion @ align_rotation 
    # initial_transform = extrinsic_0_4x4 @ opengl_conversion @ align_rotation
    print("Applying OpenGL coordinate alignment transformation...")

    S = world_points.shape[0]
    print(f"Saving point cloud video to {output_path}...")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {S}, Mode: {mode}")

    # Collect all points for setting camera view (apply transform)
    all_valid_points = []
    for i in range(S):
        pts = world_points[i].reshape(-1, 3)
        conf = depth_conf[i].reshape(-1) if depth_conf is not None else np.ones(len(pts))
        valid = np.isfinite(pts).all(axis=1) & np.any(pts != 0, axis=1) & (conf > conf_threshold)
        if np.sum(valid) > 0:
            pts_valid = pts[valid]
            # Apply transformation: pts_aligned = pts @ R.T + t
            pts_aligned = pts_valid @ initial_transform[:3, :3].T + initial_transform[:3, 3]
            if downsample_factor > 1:
                pts_aligned = pts_aligned[::downsample_factor]
            all_valid_points.append(pts_aligned)

    if len(all_valid_points) == 0:
        print("Error: No valid points found")
        return

    all_points = np.concatenate(all_valid_points, axis=0)

    # Compute scene bounds for camera setup
    center = np.mean(all_points, axis=0)
    extent = np.percentile(np.abs(all_points - center), 95) * 1.2

    temp_dir = tempfile.mkdtemp(prefix="pointcloud_video_")

    # Create renderer once (more efficient than creating per-frame)
    print("Initializing Open3D OffscreenRenderer...")
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Setup material for point rendering
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0  # Set point size in material

    # Camera setup parameters
    fov_deg = 60  # Field of view for rendering

    try:
        accumulated_points = []
        accumulated_colors = []

        for i in tqdm(range(S), desc="Rendering frames"):
            pts = world_points[i].reshape(-1, 3)
            conf = depth_conf[i].reshape(-1) if depth_conf is not None else np.ones(len(pts))
            colors = images[i].reshape(-1, 3) if images is not None else np.ones((len(pts), 3))

            valid = np.isfinite(pts).all(axis=1) & np.any(pts != 0, axis=1) & (conf > conf_threshold)

            if np.sum(valid) > 0:
                pts_valid = pts[valid]
                colors_valid = colors[valid]

                # Apply OpenGL transformation: pts_aligned = pts @ R.T + t
                pts_aligned = pts_valid @ initial_transform[:3, :3].T + initial_transform[:3, 3]

                # Downsample
                if downsample_factor > 1:
                    indices = np.arange(0, len(pts_aligned), downsample_factor)
                    pts_aligned = pts_aligned[indices]
                    colors_valid = colors_valid[indices]

                if mode == "accumulate":
                    accumulated_points.append(pts_aligned)
                    accumulated_colors.append(colors_valid)
                else:  # single mode
                    accumulated_points = [pts_aligned]
                    accumulated_colors = [colors_valid]

            # Combine all accumulated points
            if len(accumulated_points) > 0:
                all_pts = np.concatenate(accumulated_points, axis=0)
                all_cols = np.concatenate(accumulated_colors, axis=0)

                # Create point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                pcd.colors = o3d.utility.Vector3dVector(all_cols)

                # Use actual camera trajectory for view
                # extrinsics is c2w (camera to world) - already converted in postprocess()
                #   [:3, 3] = camera position in world coords (translation)
                #   [:3, 0] = camera X axis (right direction) in world
                #   [:3, 1] = camera Y axis (up direction) in world
                #   [:3, 2] = camera Z axis in world (points forward in camera convention!)
                #
                # IMPORTANT: In OpenCV camera convention, camera looks toward +Z
                # So forward (viewing) direction = +column 2 = +Z axis (the [:3, 2] column)
                cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics[i][None])[0]                                                         
                cam_position_w = cam_to_world_extrinsic[:3, 3]      # Position = translation part                                               
                cam_forward_w = -cam_to_world_extrinsic[:3, 2]      # Forward = -Z axis (camera viewing direction)                              
                cam_up_w = -cam_to_world_extrinsic[:3, 1]            # Up = Y axis  


                # Apply OpenGL transformation to camera position and directions
                # For point cloud: pts' = pts @ R.T + t (row vector form)
                # For camera position (column vector): pos' = R @ pos + t
                R_gl = initial_transform[:3, :3]
                t_gl = initial_transform[:3, 3]

                cam_position = cam_position_w @ R_gl.T + t_gl
                cam_forward = cam_forward_w @ R_gl.T 
                cam_forward = cam_forward / np.linalg.norm(cam_forward)
                cam_up = cam_up_w @ R_gl.T 
                cam_up = cam_up / np.linalg.norm(cam_up)

                # Lookat: camera should look in its forward direction (viewing direction)
                # Use a point ahead of the camera along the forward direction
                # lookat_distance = extent * 1.5  # Look some distance ahead
                # lookat_point = cam_position + cam_forward * lookat_distance 
                lookat_point = np.mean(accumulated_points[-1], axis=0) if len(accumulated_points) > 0 else cam_position + cam_forward * extent

                # Clear previous geometry and add new one
                renderer.scene.clear_geometry()
                renderer.scene.add_geometry("points", pcd, mat)

                # Setup camera following actual trajectory
                eye_position = np.array(cam_position, dtype=np.float32)
                center_lookat = np.array(lookat_point, dtype=np.float32)
                up_vector = np.array(cam_up, dtype=np.float32)
                renderer.setup_camera(fov_deg, center_lookat, eye_position, up_vector)

                # Capture image
                image = renderer.render_to_image()
                frame = np.asarray(image)
                # image is RGB uint8, convert to BGR for cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame_bgr)

            else:
                # No valid points, save black frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)

        print("Encoding video with ffmpeg...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Point cloud video saved successfully to {output_path}")
        else:
            print(f"FFmpeg error: {result.stderr}")

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up renderer
        renderer = None
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up")


def load_video_data(data_dir):
    """Load exported video data from npy files.

    Returns:
        predictions: dict with 'extrinsic', 'intrinsic', 'depth', 'depth_conf'
        images: numpy array (S, H, W, 3)
    """
    extrinsic_path = os.path.join(data_dir, "extrinsic.npy")
    intrinsic_path = os.path.join(data_dir, "intrinsic.npy")
    depth_path = os.path.join(data_dir, "depth.npy")
    depth_conf_path = os.path.join(data_dir, "depth_conf.npy")
    images_path = os.path.join(data_dir, "images.npy")

    # Check all files exist
    required_files = [extrinsic_path, intrinsic_path, depth_path, images_path]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # Load data
    extrinsic = np.load(extrinsic_path)
    intrinsic = np.load(intrinsic_path)
    depth = np.load(depth_path)
    images = np.load(images_path)

    # depth_conf is optional
    if os.path.exists(depth_conf_path):
        depth_conf = np.load(depth_conf_path)
    else:
        print("Warning: depth_conf.npy not found, using default confidence")
        depth_conf = np.ones(depth.shape[:3])

    # Build predictions dict
    predictions = {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": depth,
        "depth_conf": depth_conf,
    }

    print(f"Loaded data from {data_dir}:")
    print(f"  extrinsic: {extrinsic.shape}")
    print(f"  intrinsic: {intrinsic.shape}")
    print(f"  depth: {depth.shape}")
    print(f"  depth_conf: {depth_conf.shape}")
    print(f"  images: {images.shape}")

    return predictions, images


def main():
    parser = argparse.ArgumentParser(description="Test save_pointcloud_video_offline function")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing exported video data (extrinsic.npy, etc.)")
    parser.add_argument("--output_path", type=str, default="test_pointcloud.mp4",
                        help="Output video path (default: test_pointcloud.mp4)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video frame rate (default: 10)")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                        choices=["1280x720", "1920x1080", "3840x2160"],
                        help="Video resolution (default: 1920x1080)")
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="Confidence threshold for filtering points (default: 1.5)")
    parser.add_argument("--downsample_factor", type=int, default=10,
                        help="Downsample factor (default: 10)")
    parser.add_argument("--mode", type=str, default="accumulate",
                        choices=["accumulate", "single"],
                        help="Point cloud video mode (default: accumulate)")

    args = parser.parse_args()

    # Load data
    predictions, images = load_video_data(args.data_dir)

    # Run save_pointcloud_video_offline
    print(f"\nTesting save_pointcloud_video_offline...")
    print(f"  Output: {args.output_path}")
    print(f"  FPS: {args.fps}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"  Downsample factor: {args.downsample_factor}")
    print(f"  Mode: {args.mode}")

    save_pointcloud_video_offline(
        predictions, images, args.output_path,
        fps=args.fps, resolution=args.resolution,
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        mode=args.mode
    )

    print(f"\nTest complete! Video saved to: {args.output_path}")


if __name__ == "__main__":
    main()