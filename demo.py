"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.kv_cache_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=1,
        help="Streaming only. Every N-th frame after scale frames is kept as a keyframe. 1 = every frame.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--kv_cache_scale_frames", type=int, default=8)
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")

    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16, help="Overlap between windows")


    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")
    parser.add_argument("--export_ply", type=str, default=None,
                        help="Export 3D point cloud as PLY file after inference (no viewer needed)")
    parser.add_argument("--ply_conf_threshold", type=float, default=1.5,
                        help="Confidence threshold for PLY export (higher = fewer, higher-quality points)")
    parser.add_argument("--ply_downsample", type=int, default=10,
                        help="Downsample factor for PLY export (1 = all points, 10 = every 10th point)")
    parser.add_argument("--ply_raw", action="store_true",
                        help="Export raw points without filtering (full point cloud, large file)")
    parser.add_argument("--ply_use_depth", action="store_true", default=True,
                        help="Use depth-based points (same as viewer). False: use model world_points.")
    parser.add_argument("--export_depth_pose", type=str, default=None,
                        help="Export depth and camera poses to specified directory (npy for depth, txt for poses)")
    parser.add_argument("--no_viewer", action="store_true",
                        help="Skip launching visualization viewer (useful for batch processing)")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")

    if args.mode != "streaming" and args.keyframe_interval != 1:
        print("Warning: --keyframe_interval only applies to --mode streaming. Ignoring it for windowed inference.")
        args.keyframe_interval = 1
    elif args.mode == "streaming" and args.keyframe_interval > 1:
        print(
            f"Keyframe streaming enabled: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Inference ────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
            )

    t_infer = time.time() - t0
    print(f"Inference done: {t_infer:.1f}s ({num_frames / t_infer:.1f} FPS)")

    # ── Post-process ─────────────────────────────────────────────────────────
    predictions, images_cpu = postprocess(predictions, images)

    # ── Export PLY (if requested) ─────────────────────────────────────────────
    if args.export_ply:
        use_filtering = not args.ply_raw
        print(f"Exporting point cloud to {args.export_ply}...")
        export_raw_ply(
            predictions, images_cpu, args.export_ply,
            conf_threshold=args.ply_conf_threshold if use_filtering else 0,
            downsample_factor=args.ply_downsample if use_filtering else 1,
            use_depth=args.ply_use_depth,
        )

    # ── Export depth and poses (if requested) ─────────────────────────────────
    if args.export_depth_pose:
        export_depth_and_pose(predictions, args.export_depth_pose, images_cpu)

    # ── Visualize ────────────────────────────────────────────────────────────
    if args.no_viewer:
        print("Skipping viewer (--no_viewer mode).")
        print(f"Predictions contain keys: {list(predictions.keys())}")
        return

    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


def export_depth_and_pose(predictions, output_dir, images=None):
    """Export depth maps and camera poses to files.

    Saves:
    - depth/: directory containing S npy files, each (H, W) depth map
    - rgb/: directory containing S jpg files, each (H, W, 3) RGB image (optional)
    - extrinsic.txt: S lines, each line has 12 numbers (3x4 matrix flattened)
    - intrinsic.txt: S lines, each line has 9 numbers (3x3 matrix flattened)
    - depth_conf/: directory containing S npy files, each (H, W) confidence (optional)

    Args:
        predictions: Dictionary containing 'depth', 'extrinsic', 'intrinsic'
        output_dir: Directory to save the files
        images: Optional preprocessed images (S, 3, H, W) or (S, H, W, 3) to save as jpg
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    depth = predictions.get("depth")
    extrinsic = predictions.get("extrinsic")
    intrinsic = predictions.get("intrinsic")
    depth_conf = predictions.get("depth_conf")

    # Check required data
    if depth is None:
        print("Error: 'depth' not found in predictions")
        return
    if extrinsic is None:
        print("Error: 'extrinsic' not found in predictions")
        return
    if intrinsic is None:
        print("Error: 'intrinsic' not found in predictions")
        return

    # Convert to numpy if needed
    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.numpy()
    if depth_conf is not None and isinstance(depth_conf, torch.Tensor):
        depth_conf = depth_conf.numpy()
    if images is not None and isinstance(images, torch.Tensor):
        images = images.numpy()

    # Remove last dimension if present (H, W, 1) -> (H, W)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    S = depth.shape[0]
    H, W = depth.shape[1], depth.shape[2]
    print(f"Exporting depth and poses for {S} frames to {output_dir}")
    print(f"  Depth shape per frame: ({H}, {W})")

    # Save depth maps as individual npy files in depth/ subdirectory
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for i in range(S):
        depth_path = os.path.join(depth_dir, f"{i:06d}.npy")
        np.save(depth_path, depth[i])
    print(f"  Saved depth maps: {depth_dir}/ ({S} files)")

    # Save RGB images as jpg files in rgb/ subdirectory (if provided)
    if images is not None:
        # Handle different image formats
        if images.ndim == 4 and images.shape[1] == 3:
            # (S, 3, H, W) -> (S, H, W, 3)
            images = images.transpose(0, 2, 3, 1)

        rgb_dir = os.path.join(output_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        for i in range(S):
            img = images[i]
            # Convert from float [0,1] to uint8 [0,255]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rgb_path = os.path.join(rgb_dir, f"{i:06d}.jpg")
            cv2.imwrite(rgb_path, img_bgr)
        print(f"  Saved RGB images: {rgb_dir}/ ({S} files, shape: ({H}, {W}, 3))")

    # Save depth_conf as individual npy files (if available)
    if depth_conf is not None:
        if depth_conf.ndim == 3 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)
        conf_dir = os.path.join(output_dir, "depth_conf")
        os.makedirs(conf_dir, exist_ok=True)
        for i in range(S):
            conf_path = os.path.join(conf_dir, f"{i:06d}.npy")
            np.save(conf_path, depth_conf[i])
        print(f"  Saved depth_conf: {conf_dir}/ ({S} files)")

    # Save extrinsic as txt: S lines, each line 12 numbers
    extrinsic_path = os.path.join(output_dir, "extrinsic.txt")
    with open(extrinsic_path, 'w') as f:
        for i in range(S):
            # Flatten 3x4 to 12 numbers
            flat = extrinsic[i].reshape(12)
            f.write(" ".join([f"{v:.6f}" for v in flat]) + "\n")
    print(f"  Saved extrinsic: {extrinsic_path} ({S} lines, 12 numbers each)")

    # Save intrinsic as txt: S lines, each line 9 numbers
    intrinsic_path = os.path.join(output_dir, "intrinsic.txt")
    with open(intrinsic_path, 'w') as f:
        for i in range(S):
            # Flatten 3x3 to 9 numbers
            flat = intrinsic[i].reshape(9)
            f.write(" ".join([f"{v:.6f}" for v in flat]) + "\n")
    print(f"  Saved intrinsic: {intrinsic_path} ({S} lines, 9 numbers each)")

    print(f"Depth and pose export complete!")


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


if __name__ == "__main__":
    main()
