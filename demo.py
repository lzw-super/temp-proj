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
import logging
import os
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

# Configure logging to show INFO level messages both in terminal and log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler('demo.log', mode='w'),  # Output to log file
    ]
)

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general , closed_form_inverse_se3
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8,
                test_resolution=None):
    """Load images from folder or video and preprocess into a tensor.

    Args:
        test_resolution: Optional test resolution override ("240p", "360p", "480p").
                         If set, images are resized to this resolution instead of
                         the default crop/pad preprocessing.

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
        test_resolution=test_resolution,
    )
    h, w = images.shape[-2:]
    if test_resolution is not None:
        print(f"Preprocessed images to {w}x{h} (test resolution: {test_resolution})")
    else:
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
        img_size=518,  #固定的值
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
    parser.add_argument("--test_resolution", type=str, default=None, choices=["240p", "360p", "480p"],
                        help="Test processing time with specified input resolution (240p=308x238, 360p=476x350, 480p=630x476)")

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

    # Video saving options
    parser.add_argument("--save_video", type=str, default=None,
                        help="Save videos to specified path. "
                             "Options: 'auto' (save to dataset folder), or specify path (e.g., output). "
                             "Generates: *_pointcloud.mp4 (point cloud animation) "
                             "and *_original.mp4 (original images). "
                             "In --no_viewer mode: always saves both videos.")
    parser.add_argument("--video_fps", type=int, default=10,
                        help="Video frame rate (default: 10)")
    parser.add_argument("--video_resolution", type=str, default="1920x1080",
                        choices=["1280x720", "1920x1080", "3840x2160"],
                        help="Video resolution (default: 1920x1080)")
    parser.add_argument("--save_pointcloud_video", action="store_true", default=False,
                        help="Force save point cloud video even in viewer mode "
                             "(uses Open3D offline rendering, GPU accelerated)")
    parser.add_argument("--video_mode", type=str, default="accumulate",
                        choices=["accumulate", "single"],
                        help="Point cloud video mode: 'accumulate' shows all frames up to current "
                             "(3D mode), 'single' shows only current frame (4D mode)")
    parser.add_argument("--export_video_data", type=str, default=None,
                        help="Export data for save_pointcloud_video_offline testing. "
                             "Specify directory path to save extrinsic.npy, intrinsic.npy, "
                             "depth.npy, depth_conf.npy, images.npy")

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
        test_resolution=args.test_resolution,
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

    # ── Export test RGB images (if test_resolution is set) ───────────────────────
    if args.test_resolution is not None:
        test_output_dir = os.path.join(os.path.dirname(args.image_folder) if args.image_folder else '.', 'test')
        os.makedirs(test_output_dir, exist_ok=True)
        print(f"Exporting test RGB images to {test_output_dir}/ (resolution: {args.test_resolution})")

        # images_cpu already contains resized images (test_resolution applied in load_and_preprocess_images)
        if isinstance(images_cpu, torch.Tensor):
            images_np = images_cpu.cpu().numpy()
        else:
            images_np = images_cpu

        # Remove batch dimension if present
        if images_np.ndim == 5 and images_np.shape[0] == 1:
            images_np = images_np[0]

        S = images_np.shape[0]
        # Convert (S, 3, H, W) -> (S, H, W, 3)
        if images_np.ndim == 4 and images_np.shape[1] == 3:
            images_np = images_np.transpose(0, 2, 3, 1)

        H, W = images_np.shape[1], images_np.shape[2]
        for i in range(S):
            img = images_np[i]
            # Convert from float [0,1] to uint8 [0,255]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(test_output_dir, f"{i:06d}.jpg")
            cv2.imwrite(img_path, img_bgr)
        print(f"  Saved {S} RGB images ({W}x{H}) to {test_output_dir}/")

    # ── Export PLY (if requested) ─────────────────────────────────────────────
    if args.export_ply :
        use_filtering = not args.ply_raw 
        print(f"Exporting point cloud to {os.path.dirname(args.image_folder)}...")
        export_raw_ply(
            predictions, images_cpu, os.path.dirname(args.image_folder),
            conf_threshold=args.ply_conf_threshold if use_filtering else 0,
            downsample_factor=args.ply_downsample if use_filtering else 1,
            use_depth=args.ply_use_depth,
        )

    # ── Export depth and poses (if requested) ─────────────────────────────────
    if args.export_depth_pose:
        export_depth_and_pose(predictions, args.export_depth_pose, images_cpu)

    # ── Export video data for testing (if requested) ───────────────────────────
    if args.export_video_data:
        export_video_data(predictions, images_cpu, args.export_video_data)

    # ── Save video (if requested) ───────────────────────────────────────────────
    if args.save_video :
        # Determine video output directory
        # If save_video is a directory or empty, use dataset directory
        if args.save_video == "auto" or args.save_video == "" or args.image_folder :
            # Auto-detect dataset directory from input source
            if args.image_folder:
                video_output_dir = os.path.dirname(args.image_folder) # 父目录
            elif args.video_path:
                video_output_dir = os.path.dirname(args.video_path)
            else:
                video_output_dir = resolved_image_folder
            # Use dataset folder name as video prefix
            dataset_name = os.path.basename(video_output_dir.rstrip('/'))
            video_base_path = os.path.join(video_output_dir, f"{dataset_name}_pointcloud.mp4")
        else:
            # Use specified path
            video_base_path = args.save_video
            if not video_base_path.endswith('.mp4'):
                video_base_path = video_base_path + '.mp4'

        # Extract base name for related videos
        base_name = os.path.splitext(video_base_path)[0]
        pointcloud_video_path = f"{base_name}_pointcloud.mp4"
        original_video_path = f"{base_name}_original.mp4"

        print(f"Videos will be saved to:")
        print(f"  Directory: {os.path.dirname(pointcloud_video_path)}")
        print(f"  Point cloud: {os.path.basename(pointcloud_video_path)}")
        print(f"  Original: {os.path.basename(original_video_path)}")

        # In no_viewer mode, always save both videos if save_video is specified
        if args.no_viewer:
            print("Saving videos in --no_viewer mode...")

            # Save point cloud video using Open3D offline rendering
            save_pointcloud_video_offline(
                predictions, images_cpu, pointcloud_video_path,
                fps=args.video_fps, resolution=args.video_resolution,
                conf_threshold=args.conf_threshold,
                downsample_factor=args.downsample_factor,
                mode=args.video_mode
            )

            # Save original video
            save_original_video(
                images_cpu, original_video_path,
                fps=args.video_fps, resolution=args.video_resolution
            )

            print(f"Videos saved:")
            print(f"  Point cloud video: {pointcloud_video_path}")
            print(f"  Original video: {original_video_path}")

        elif args.save_pointcloud_video:
            # In viewer mode, save point cloud video on request
            save_pointcloud_video_offline(
                predictions, images_cpu, pointcloud_video_path,
                fps=args.video_fps, resolution=args.video_resolution,
                conf_threshold=args.conf_threshold,
                downsample_factor=args.downsample_factor,
                mode=args.video_mode
            )
            # Also save original video
            save_original_video(
                images_cpu, original_video_path,
                fps=args.video_fps, resolution=args.video_resolution
            )
        else:
            # Only save original video (no point cloud)
            save_original_video(
                images_cpu, original_video_path,
                fps=args.video_fps, resolution=args.video_resolution
            )
        print("Video saving complete.")

    # ── Skip viewer if requested ────────────────────────────────────────────────
    if args.no_viewer:
        print("Skipping viewer (--no_viewer mode).")
        print(f"Predictions contain keys: {list(predictions.keys())}")
        return

    # ── Visualize ────────────────────────────────────────────────────────────────
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
    from lingbot_map.utils.geometry import closed_form_inverse_se3,closed_form_inverse_se3_general  #取逆 w2c转c2w

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
    # cam_to_world_extrinsic = closed_form_inverse_se3_general(extrinsic)
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

    extrinsic_4x4 = np.eye(4)[None].repeat(S, axis=0)  # (S, 4, 4)
    extrinsic_4x4[:, :3, :4] = extrinsic

    # 批量求逆：c2w → w2c
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic_4x4)
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
            # Flatten 4x4 to 16 numbers
            flat = cam_to_world_extrinsic[i].reshape(16)
            f.write(" ".join([f"{v:.12f}" for v in flat]) + "\n")
    print(f"  Saved extrinsic: {extrinsic_path} ({S} lines, 16 numbers each)")

    # Save intrinsic as txt: S lines, each line 9 numbers
    intrinsic_path = os.path.join(output_dir, "intrinsic.txt")
    with open(intrinsic_path, 'w') as f:
        for i in range(S):
            # Flatten 3x3 to 9 numbers
            flat = intrinsic[i].reshape(9)
            f.write(" ".join([f"{v:.6f}" for v in flat]) + "\n")
    print(f"  Saved intrinsic: {intrinsic_path} ({S} lines, 9 numbers each)")

    print(f"Depth and pose export complete!")


def export_video_data(predictions, images, output_dir):
    """Export data needed for save_pointcloud_video_offline to npy files.

    Saves:
    - extrinsic.npy: (S, 3, 4) camera extrinsics (c2w)
    - intrinsic.npy: (S, 3, 3) camera intrinsics
    - depth.npy: (S, H, W, 1) depth maps
    - depth_conf.npy: (S, H, W) depth confidence
    - images.npy: (S, H, W, 3) RGB images

    Args:
        predictions: Dictionary containing 'extrinsic', 'intrinsic', 'depth', 'depth_conf'
        images: Image tensor (S, 3, H, W) or (S, H, W, 3)
        output_dir: Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)

    extrinsics = predictions.get("extrinsic")
    intrinsics = predictions.get("intrinsic")
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")

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

    # Handle batch dimension for images
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]

    # Convert (S, 3, H, W) -> (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)

    S = depth.shape[0]
    print(f"Exporting video data for {S} frames to {output_dir}")

    # Save each array
    np.save(os.path.join(output_dir, "extrinsic.npy"), extrinsics)
    np.save(os.path.join(output_dir, "intrinsic.npy"), intrinsics)
    np.save(os.path.join(output_dir, "depth.npy"), depth)
    np.save(os.path.join(output_dir, "depth_conf.npy"), depth_conf)
    np.save(os.path.join(output_dir, "images.npy"), images)

    print(f"  extrinsic.npy: {extrinsics.shape}")
    print(f"  intrinsic.npy: {intrinsics.shape}")
    print(f"  depth.npy: {depth.shape}")
    print(f"  depth_conf.npy: {depth_conf.shape}")
    print(f"  images.npy: {images.shape}")
    print(f"Video data export complete!")


def save_original_video(images, output_path, fps=30, resolution="1920x1080"):
    """Save original images as a video file.

    Args:
        images: Image tensor or numpy array (S, 3, H, W) or (S, H, W, 3)
        output_path: Path to save the video (e.g., output.mp4)
        fps: Frame rate (default: 30)
        resolution: Resolution string (e.g., "1920x1080")
    """
    import subprocess
    import tempfile
    import shutil

    width, height = map(int, resolution.split('x'))

    # Convert to numpy if needed
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    # Handle batch dimension
    if images_np.ndim == 5 and images_np.shape[0] == 1:
        images_np = images_np[0]

    S = images_np.shape[0]

    # Convert (S, 3, H, W) -> (S, H, W, 3)
    if images_np.ndim == 4 and images_np.shape[1] == 3:
        images_np = images_np.transpose(0, 2, 3, 1)

    print(f"Saving original video to {output_path}...")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {S}")

    temp_dir = tempfile.mkdtemp(prefix="original_video_")

    try:
        for i in tqdm(range(S), desc="Preparing frames"):
            img = images_np[i]
            # Convert from float [0,1] to uint8 [0,255]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # Resize to target resolution
            img_resized = cv2.resize(img, (width, height))
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, img_bgr)

        print("Encoding video with ffmpeg...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Original video saved successfully to {output_path}")
        else:
            print(f"FFmpeg error: {result.stderr}")

    finally:
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up")


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
    mat.point_size = 2.0  # Set point size in material

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
                #
                # extrinsics is c2w (camera-to-world) - converted from w2c in postprocess()
                # c2w 表示相机在世界坐标系中的位置和姿态:
                #   [:3, 3] = 相机在世界坐标系中的位置 (translation)
                #   [:3, 0] = 相机 X轴在世界坐标系中的方向 (right)
                #   [:3, 1] = 相机 Y轴在世界坐标系中的方向 (up/down in OpenCV)
                #   [:3, 2] = 相机 Z轴在世界坐标系中的方向 (forward in OpenCV)
                #
                # 对于渲染，需要 w2c (world-to-camera) 视角矩阵:
                # w2c = inv(c2w) 描述"世界如何投影到相机"
                # Open3D setup_camera 需要相机视角信息（eye, lookat, up）
                # 这些信息从 w2c 矩阵中提取更直接
                #
                # w2c 的结构（对 c2w 求逆后）:
                #   [:3, 3] = -R^T @ t (世界原点在相机坐标系中的位置)
                #   [:3, 2] = R的第3行 = 相机Z轴在世界坐标系中的方向 (forward)
                #   [:3, 1] = R的第2行 = 相机Y轴在世界坐标系中的方向 (up)
                #
                # IMPORTANT: 在 OpenGL/Open3D 渲染中:
                #   - forward = 相机朝向方向（看向场景）
                #   - 在 OpenCV 约定中，相机看向 +Z
                #   - 所以 forward = [:3, 2] (Z轴方向)
                #   - up = [:3, 1] (Y轴方向，OpenCV中向下，需要取负)

                # 将 c2w 转换为 w2c (求逆)
                cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics[i][None])[0]
                # 从 w2c 矩阵提取相机视角信息
                cam_position_w = cam_to_world_extrinsic[:3, 3]       # 相机位置（从w2c提取）
                cam_forward_w = -cam_to_world_extrinsic[:3, 2]       # Forward方向（相机看向的方向）
                cam_up_w = -cam_to_world_extrinsic[:3, 1]            # Up方向（取负因为在OpenCV中Y向下）  


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
