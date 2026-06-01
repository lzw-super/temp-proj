"""
Validate the scale relationship between demo.py depth predictions and Replica GT.

This script mirrors demo.py inference (GCTStream.inference_streaming/windowed) and
compares the raw predicted depth against Replica metric GT depth. It also tests
whether multiplying the raw prediction by a GT-derived anchor scale makes it
closer to metric depth.

Typical use:
    python try_train/validate_demo_depth_scale.py

Useful quick run:
    python try_train/validate_demo_depth_scale.py --first_k 8 --use_sdpa
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "try_train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "try_train"))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from replica_dataset import REPLICA_DEPTH_SCALE


def _frame_id(path: Path) -> int:
    match = re.search(r"frame(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse Replica frame id from {path}")
    return int(match.group(1))


def resolve_results_dir(data_root: Path) -> Path:
    if (data_root / "results").is_dir():
        return data_root / "results"
    if list(data_root.glob("frame*.jpg")):
        return data_root
    raise FileNotFoundError(f"Cannot find Replica results directory under {data_root}")


def collect_frame_paths(data_root: Path, first_k: int, stride: int, start: int):
    results_dir = resolve_results_dir(data_root)
    paths = sorted(results_dir.glob("frame*.jpg"))
    paths = paths[start::stride]
    if first_k > 0:
        paths = paths[:first_k]
    if not paths:
        raise RuntimeError(f"No frames found in {results_dir}")
    for path in paths:
        fid = _frame_id(path)
        depth_path = results_dir / f"depth{fid:06d}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth file for {path}: {depth_path}")
    return paths


def load_replica_poses(data_root: Path):
    traj_path = data_root / "traj.txt"
    if not traj_path.exists():
        traj_path = data_root.parent / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Cannot find traj.txt for {data_root}")

    poses = {}
    with open(traj_path, "r") as f:
        for idx, line in enumerate(f):
            vals = [float(v) for v in line.strip().split()]
            if len(vals) == 16:
                poses[idx] = np.asarray(vals, dtype=np.float32).reshape(4, 4)
    return poses


def _target_resolution(test_resolution):
    if test_resolution is None:
        return None
    mapping = {
        "240p": (308, 238),
        "360p": (476, 350),
        "480p": (630, 476),
    }
    return mapping[test_resolution]


def preprocess_depth_like_demo(
    depth_path: Path,
    rgb_path: Path,
    image_size: int,
    patch_size: int,
    mode: str,
    test_resolution: str | None,
):
    """Apply the same resize/crop/pad geometry as load_and_preprocess_images."""
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    with Image.open(rgb_path) as rgb_img:
        width, height = rgb_img.size

    fx = fy = 600.0
    cx = width / 2.0
    cy = height / 2.0

    test_hw = _target_resolution(test_resolution)
    if test_hw is not None:
        new_width, new_height = test_hw
    elif mode == "pad":
        if width >= height:
            new_width = image_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = image_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
    elif mode == "crop":
        new_width = image_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    with Image.open(depth_path) as depth_img:
        depth_img = depth_img.resize((new_width, new_height), Image.Resampling.NEAREST)
        depth = np.asarray(depth_img).astype(np.float32) / REPLICA_DEPTH_SCALE
    sx = new_width / width
    sy = new_height / height
    K = np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    if test_resolution is None and mode == "crop" and new_height > image_size:
        start_y = (new_height - image_size) // 2
        depth = depth[start_y : start_y + image_size, :]
        K[1, 2] -= start_y

    if test_resolution is None and mode == "pad":
        h_padding = image_size - depth.shape[0]
        w_padding = image_size - depth.shape[1]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            depth = np.pad(
                depth,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            K[0, 2] += pad_left
            K[1, 2] += pad_top

    return depth, K


def pad_to_common_shape(depths, intrinsics):
    max_h = max(d.shape[0] for d in depths)
    max_w = max(d.shape[1] for d in depths)
    out_depths = []
    out_intrinsics = []
    for depth, K in zip(depths, intrinsics):
        h, w = depth.shape
        pad_h = max_h - h
        pad_w = max_w - w
        K = K.copy()
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            depth = np.pad(
                depth,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            K[0, 2] += pad_left
            K[1, 2] += pad_top
        out_depths.append(depth)
        out_intrinsics.append(K)
    return np.stack(out_depths, axis=0), np.stack(out_intrinsics, axis=0)


def load_gt_depths_and_intrinsics(frame_paths, image_size, patch_size, mode, test_resolution):
    depths = []
    intrinsics = []
    for rgb_path in frame_paths:
        fid = _frame_id(rgb_path)
        depth_path = rgb_path.parent / f"depth{fid:06d}.png"
        depth, K = preprocess_depth_like_demo(
            depth_path,
            rgb_path,
            image_size=image_size,
            patch_size=patch_size,
            mode=mode,
            test_resolution=test_resolution,
        )
        depths.append(depth)
        intrinsics.append(K)
    depths, intrinsics = pad_to_common_shape(depths, intrinsics)
    masks = np.isfinite(depths) & (depths > 0)
    return depths.astype(np.float32), masks, intrinsics.astype(np.float32)


def load_relative_poses(data_root: Path, frame_paths):
    poses_by_id = load_replica_poses(data_root)
    raw = []
    for path in frame_paths:
        fid = _frame_id(path)
        if fid not in poses_by_id:
            raise KeyError(f"Frame {fid} not found in traj.txt")
        raw.append(poses_by_id[fid])
    ref_inv = np.linalg.inv(raw[0])
    return np.stack([ref_inv @ p for p in raw], axis=0).astype(np.float32)


def gt_anchor_depth_median(depths, masks, num_anchor_frames):
    n = min(num_anchor_frames, depths.shape[0])
    valid = depths[:n][masks[:n]]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def gt_anchor_pointcloud_mean_scale(
    depths,
    masks,
    intrinsics,
    poses_c2w_rel,
    num_anchor_frames,
    sample_stride,
):
    n = min(num_anchor_frames, depths.shape[0])
    total = 0.0
    count = 0
    for i in range(n):
        depth = depths[i, ::sample_stride, ::sample_stride]
        mask = masks[i, ::sample_stride, ::sample_stride]
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs]
        K = intrinsics[i]
        x = (xs * sample_stride - K[0, 2]) / K[0, 0] * z
        y = (ys * sample_stride - K[1, 2]) / K[1, 1] * z
        pts_cam = np.stack([x, y, z], axis=1)
        R = poses_c2w_rel[i, :3, :3]
        t = poses_c2w_rel[i, :3, 3]
        pts_ref = pts_cam @ R.T + t
        norms = np.linalg.norm(pts_ref, axis=1)
        total += float(norms.sum())
        count += int(norms.size)
    if count == 0:
        return float("nan")
    return total / count


def build_model(args, device):
    if args.mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=518,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.kv_cache_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
    )

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model", "full_model_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                print(f"[load] using checkpoint key: {key}")
                break
    if state_dict is None:
        state_dict = ckpt

    if all(k.startswith("module.") for k in list(state_dict.keys())[:20]):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] checkpoint: {args.model_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[load] first missing keys: {missing[:5]}")
    if unexpected:
        print(f"[load] first unexpected keys: {unexpected[:5]}")
    return model.to(device).eval()


def run_demo_inference(model, images, args, device):
    images = images.to(device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = torch.amp.autocast("cpu", enabled=False)

    with torch.no_grad(), autocast_ctx:
        if args.mode == "streaming":
            pred = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
        else:
            pred = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
    depth = pred["depth"]
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth.detach().float().cpu().numpy()


def masked_values(pred, gt, mask):
    ok = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    return pred[ok].astype(np.float64), gt[ok].astype(np.float64), ok


def depth_metrics(pred, gt, mask):
    pred_v, gt_v, ok = masked_values(pred, gt, mask)
    if pred_v.size == 0:
        return {
            "valid_pixels": 0,
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "log_l1": float("nan"),
            "pred_median": float("nan"),
            "gt_median": float("nan"),
            "median_gt_over_pred": float("nan"),
        }
    diff = pred_v - gt_v
    return {
        "valid_pixels": int(pred_v.size),
        "abs_rel": float(np.mean(np.abs(diff) / gt_v)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "log_l1": float(np.mean(np.abs(np.log(pred_v + 1e-6) - np.log(gt_v + 1e-6)))),
        "pred_median": float(np.median(pred_v)),
        "gt_median": float(np.median(gt_v)),
        "median_gt_over_pred": float(np.median(gt_v / pred_v)),
    }


def best_global_scale(pred, gt, mask):
    pred_v, gt_v, _ = masked_values(pred, gt, mask)
    if pred_v.size == 0:
        return float("nan")
    return float(np.median(gt_v / pred_v))


def resize_nearest_np(arr, out_hw):
    out_h, out_w = out_hw
    img = Image.fromarray(arr)
    return np.asarray(img.resize((out_w, out_h), Image.Resampling.NEAREST))


def classify_scale(best_scale, anchor_depth_s, anchor_point_s):
    if not np.isfinite(best_scale):
        return "Cannot classify: no valid depth overlap."
    close_to_meter = 0.8 <= best_scale <= 1.25
    close_to_depth_anchor = (
        np.isfinite(anchor_depth_s) and 0.8 <= best_scale / anchor_depth_s <= 1.25
    )
    close_to_point_anchor = (
        np.isfinite(anchor_point_s) and 0.8 <= best_scale / anchor_point_s <= 1.25
    )
    if close_to_meter:
        return "Raw demo depth is meter-like (best scale is close to 1)."
    if close_to_depth_anchor or close_to_point_anchor:
        return "Raw demo depth looks anchor-normalized (best scale is close to GT anchor_s)."
    return "Raw demo depth has a different scale; inspect best_scale and anchor_s values."


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Validate demo.py predicted depth scale against Replica GT."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/shared_files/datasets/dovsg/Replica/room0"),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("/home/shared_files/model_weights/linbo_map/lingbot-map.pt"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("try_train/demo_depth_scale_check"))
    parser.add_argument("--first_k", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--test_resolution", choices=["240p", "360p", "480p"], default=None)
    parser.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--overlap_size", type=int, default=16)
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--kv_cache_scale_frames", type=int, default=8)
    parser.add_argument(
        "--use_sdpa",
        action="store_true",
        default=True,
        help="Use SDPA backend. Default True for easier local validation.",
    )
    parser.add_argument(
        "--use_flashinfer",
        dest="use_sdpa",
        action="store_false",
        help="Use FlashInfer backend, matching demo.py default.",
    )
    parser.add_argument("--point_scale_sample_stride", type=int, default=8)
    args = parser.parse_args()

    frame_paths = collect_frame_paths(args.data_root, args.first_k, args.stride, args.start)
    frame_ids = [_frame_id(p) for p in frame_paths]
    print(f"[data] frames={len(frame_paths)}, ids={frame_ids[:5]}...{frame_ids[-3:]}")
    print(f"[data] Replica depth scale: raw / {REPLICA_DEPTH_SCALE}")

    print("[data] loading demo-preprocessed RGB images...")
    images = load_and_preprocess_images(
        [str(p) for p in frame_paths],
        mode=args.preprocess_mode,
        image_size=args.image_size,
        patch_size=args.patch_size,
        test_resolution=args.test_resolution,
    )

    print("[data] loading GT depth with the same geometry preprocessing...")
    gt_depths, gt_masks, gt_intrinsics = load_gt_depths_and_intrinsics(
        frame_paths,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mode=args.preprocess_mode,
        test_resolution=args.test_resolution,
    )
    poses_rel = load_relative_poses(args.data_root, frame_paths)

    anchor_depth_s = gt_anchor_depth_median(gt_depths, gt_masks, args.num_scale_frames)
    anchor_point_s = gt_anchor_pointcloud_mean_scale(
        gt_depths,
        gt_masks,
        gt_intrinsics,
        poses_rel,
        args.num_scale_frames,
        args.point_scale_sample_stride,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] device={device}, mode={args.mode}, use_sdpa={args.use_sdpa}")
    model = build_model(args, device)

    print("[infer] running demo-style inference...")
    pred_depth = run_demo_inference(model, images, args, device)

    if pred_depth.shape != gt_depths.shape:
        print(f"[warn] pred depth shape {pred_depth.shape} != GT {gt_depths.shape}; resizing GT.")
        resized_gt = []
        resized_mask = []
        for d, m in zip(gt_depths, gt_masks):
            resized_gt.append(resize_nearest_np(d, (pred_depth.shape[1], pred_depth.shape[2])))
            resized_mask.append(
                resize_nearest_np(m.astype(np.uint8), (pred_depth.shape[1], pred_depth.shape[2])).astype(bool)
            )
        gt_depths = np.stack(resized_gt, axis=0)
        gt_masks = np.stack(resized_mask, axis=0)

    best_s = best_global_scale(pred_depth, gt_depths, gt_masks)

    summary = {
        "config": {
            "data_root": str(args.data_root),
            "model_path": str(args.model_path),
            "frame_ids": frame_ids,
            "preprocess_mode": args.preprocess_mode,
            "image_size": args.image_size,
            "test_resolution": args.test_resolution,
            "mode": args.mode,
            "num_scale_frames": args.num_scale_frames,
            "replica_depth_scale": REPLICA_DEPTH_SCALE,
        },
        "shapes": {
            "images": list(images.shape),
            "pred_depth": list(pred_depth.shape),
            "gt_depth": list(gt_depths.shape),
        },
        "gt_anchor_scales": {
            "depth_median_proxy": anchor_depth_s,
            "paper_pointcloud_mean": anchor_point_s,
        },
        "global_scale_fit": {
            "best_median_gt_over_pred": best_s,
            "best_over_depth_anchor_s": float(best_s / anchor_depth_s) if np.isfinite(anchor_depth_s) else float("nan"),
            "best_over_point_anchor_s": float(best_s / anchor_point_s) if np.isfinite(anchor_point_s) else float("nan"),
        },
        "metrics": {
            "raw_pred_vs_gt_m": depth_metrics(pred_depth, gt_depths, gt_masks),
            "pred_times_best_scale_vs_gt_m": depth_metrics(pred_depth * best_s, gt_depths, gt_masks),
            "pred_times_depth_anchor_s_vs_gt_m": depth_metrics(pred_depth * anchor_depth_s, gt_depths, gt_masks),
            "pred_times_point_anchor_s_vs_gt_m": depth_metrics(pred_depth * anchor_point_s, gt_depths, gt_masks),
        },
    }
    summary["interpretation"] = classify_scale(best_s, anchor_depth_s, anchor_point_s)

    rows = []
    for idx, fid in enumerate(frame_ids):
        raw = depth_metrics(pred_depth[idx], gt_depths[idx], gt_masks[idx])
        rows.append(
            {
                "frame_idx": idx,
                "replica_frame_id": fid,
                "pred_median_raw": raw["pred_median"],
                "gt_median_m": raw["gt_median"],
                "median_gt_over_pred": raw["median_gt_over_pred"],
                "raw_abs_rel": raw["abs_rel"],
                "raw_rmse": raw["rmse"],
                "raw_log_l1": raw["log_l1"],
                "best_scaled_abs_rel": depth_metrics(
                    pred_depth[idx] * best_s, gt_depths[idx], gt_masks[idx]
                )["abs_rel"],
                "depth_anchor_scaled_abs_rel": depth_metrics(
                    pred_depth[idx] * anchor_depth_s, gt_depths[idx], gt_masks[idx]
                )["abs_rel"],
                "point_anchor_scaled_abs_rel": depth_metrics(
                    pred_depth[idx] * anchor_point_s, gt_depths[idx], gt_masks[idx]
                )["abs_rel"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "per_frame_scale.csv"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    save_csv(csv_path, rows)

    print("\n=== Depth Scale Summary ===")
    print(f"Frames: {len(frame_ids)}  pred_shape={pred_depth.shape}")
    print(f"GT anchor_s depth median proxy: {anchor_depth_s:.6f} m")
    print(f"GT anchor_s paper point-cloud mean: {anchor_point_s:.6f} m")
    print(f"Best global scale median(gt / pred_raw): {best_s:.6f}")
    print(f"best / depth_anchor_s: {summary['global_scale_fit']['best_over_depth_anchor_s']:.6f}")
    print(f"best / point_anchor_s: {summary['global_scale_fit']['best_over_point_anchor_s']:.6f}")
    print(f"Interpretation: {summary['interpretation']}")
    print("\nMetrics (AbsRel / RMSE / LogL1):")
    for name, m in summary["metrics"].items():
        print(f"  {name}: {m['abs_rel']:.6f} / {m['rmse']:.6f} / {m['log_l1']:.6f}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
