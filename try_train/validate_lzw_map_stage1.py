"""Validate LZW-Map Stage1 against initialization, original LingBot-Map, and GT."""

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from replica_dataset import ReplicaDataset
from lingbot_map.models.lzw_map import create_lzw_map_stage1
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.rotation import quat_to_mat
from train_aggregator_only import forward_aggregator_heads, load_depth_gct_model
from train_lzw_map_stage1 import (
    DEFAULT_DINOV2_REPO,
    forward_lzw_stage1,
    load_backbone_from_original_dinov2_repo,
)


POSE_AUC_THRESHOLDS = (3, 5, 15, 30)


def autocast_context(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def maybe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def median_align_depth(pred, gt, mask):
    aligned = pred.clone()
    for batch_idx in range(pred.shape[0]):
        for view_idx in range(pred.shape[1]):
            valid = mask[batch_idx, view_idx].bool()
            if valid.sum() < 10:
                continue
            pred_median = pred[batch_idx, view_idx][valid].median().clamp(min=1e-6)
            gt_median = gt[batch_idx, view_idx][valid].median()
            aligned[batch_idx, view_idx] = pred[batch_idx, view_idx] * (gt_median / pred_median)
    return aligned


def scale_align_pose_translation(pose_enc, pose_gt):
    """Anchor at frame 0 and fit one positive translation scale per batch."""
    aligned = pose_enc.clone()
    pred_center = pose_enc[:, :, :3] - pose_enc[:, :1, :3]
    gt_center = pose_gt[:, :, :3, 3]

    for batch_idx in range(pose_enc.shape[0]):
        pred = pred_center[batch_idx, 1:].float()
        gt = gt_center[batch_idx, 1:].float()
        denominator = pred.square().sum().clamp(min=1e-8)
        scale = (pred * gt).sum() / denominator
        scale = scale.clamp(min=1e-6, max=1e6)
        aligned[batch_idx, :, :3] = pred_center[batch_idx] * scale.to(pred_center.dtype)
    return aligned


def normalize_quaternion(quat):
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def pose_quaternion_to_rotation(quat, convention):
    """Convert pose-encoding quaternion to rotation matrix.

    LingBot's official pose encoding uses XYZW (scalar-last). The old local
    training loss interpreted the same four slots as WXYZ, so validation reports
    both conventions explicitly.
    """
    quat = normalize_quaternion(quat.float())
    if convention == "xyzw":
        quat_xyzw = quat
    elif convention == "wxyz":
        quat_xyzw = torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)
    else:
        raise ValueError(f"Unknown quaternion convention: {convention}")
    return quat_to_mat(quat_xyzw)


def rotation_angle_rad(rot_pred, rot_gt):
    rel = rot_pred.float().transpose(-1, -2) @ rot_gt.float()
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    return torch.acos(cos_angle)


def tensor_mean(value):
    return float(value.mean().item()) if value.numel() else float("nan")


def tensor_median(value):
    return float(value.median().item()) if value.numel() else float("nan")


def tensor_max(value):
    return float(value.max().item()) if value.numel() else float("nan")


def tensor_rmse(value):
    return float(torch.sqrt(torch.mean(value.float().square())).item()) if value.numel() else float("nan")


def sim3_align_centers(pred_center, gt_center):
    """Umeyama Sim(3) alignment from predicted centers to GT centers."""
    pred_center = pred_center.float()
    gt_center = gt_center.float()
    batch_size, num_views, _ = pred_center.shape
    aligned = torch.empty_like(pred_center)
    rotations = []
    scales = []

    for batch_idx in range(batch_size):
        pred = pred_center[batch_idx]
        gt = gt_center[batch_idx]
        pred_mean = pred.mean(dim=0)
        gt_mean = gt.mean(dim=0)
        pred_zero = pred - pred_mean
        gt_zero = gt - gt_mean
        pred_var = pred_zero.square().sum() / max(num_views, 1)

        if num_views < 2 or pred_var < 1e-10:
            rotation = torch.eye(3, device=pred_center.device, dtype=pred_center.dtype)
            scale = torch.zeros((), device=pred_center.device, dtype=pred_center.dtype)
            aligned[batch_idx] = gt_mean.expand_as(pred)
        else:
            covariance = gt_zero.transpose(0, 1) @ pred_zero / num_views
            u, singular_values, vh = torch.linalg.svd(covariance)
            det = torch.det(u @ vh)
            correction = torch.ones(3, device=pred_center.device, dtype=pred_center.dtype)
            if det < 0:
                correction[-1] = -1
            rotation = u @ torch.diag(correction) @ vh
            scale = (singular_values * correction).sum() / pred_var
            scale = scale.clamp(min=1e-8)
            aligned[batch_idx] = scale * (pred_zero @ rotation.transpose(0, 1)) + gt_mean

        rotations.append(rotation)
        scales.append(scale)

    return aligned, torch.stack(scales), torch.stack(rotations)


def invert_transform(transform):
    inv = torch.empty_like(transform)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_t = rot.transpose(-1, -2)
    inv[..., :3, :3] = rot_t
    inv[..., :3, 3] = -(rot_t @ trans.unsqueeze(-1)).squeeze(-1)
    inv[..., 3, :3] = 0
    inv[..., 3, 3] = 1
    return inv


def build_pose_matrix(rotation, translation):
    pose = torch.zeros(
        rotation.shape[:-2] + (4, 4),
        device=rotation.device,
        dtype=rotation.dtype,
    )
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = translation
    pose[..., 3, 3] = 1
    return pose


def align_to_first_camera(extrinsics):
    first_inv = invert_transform(extrinsics[:, :1])
    return extrinsics @ first_inv


def translation_angle_deg(trans_pred, trans_gt):
    pred_norm = torch.norm(trans_pred, dim=-1, keepdim=True)
    gt_norm = torch.norm(trans_gt, dim=-1, keepdim=True)
    pred_unit = trans_pred / pred_norm.clamp(min=1e-15)
    gt_unit = trans_gt / gt_norm.clamp(min=1e-15)
    cos_angle = (pred_unit * gt_unit).sum(dim=-1).abs().clamp(min=-1.0, max=1.0)
    angle = torch.acos(cos_angle) * (180.0 / np.pi)
    invalid = (pred_norm.squeeze(-1) <= 1e-15) | (gt_norm.squeeze(-1) <= 1e-15)
    return torch.where(invalid, torch.full_like(angle, 90.0), angle)


def calculate_pose_auc(rot_error, trans_error, thresholds=POSE_AUC_THRESHOLDS):
    valid = torch.isfinite(rot_error) & torch.isfinite(trans_error)
    metrics = {}
    if not valid.any():
        for threshold in thresholds:
            metrics.update({
                f"auc{threshold}": float("nan"),
                f"racc{threshold}": float("nan"),
                f"tacc{threshold}": float("nan"),
            })
        return metrics

    rot_error_np = rot_error[valid].detach().cpu().numpy()
    trans_error_np = trans_error[valid].detach().cpu().numpy()
    max_errors = np.maximum(rot_error_np, trans_error_np)
    num_pairs = float(len(max_errors))

    for threshold in thresholds:
        bins = np.arange(threshold + 1)
        histogram, _ = np.histogram(max_errors, bins=bins)
        auc = float(np.mean(np.cumsum(histogram.astype(float) / num_pairs)) * 100.0)
        metrics.update({
            f"auc{threshold}": auc,
            f"racc{threshold}": float(np.mean(rot_error_np < threshold) * 100.0),
            f"tacc{threshold}": float(np.mean(trans_error_np < threshold) * 100.0),
        })
    return metrics


def compute_scale_anchor_info(pose_enc, pose_gt):
    pred_center = pose_enc[:, :, :3].float()
    gt_center = pose_gt[:, :, :3, 3].float()
    pred_anchor = pred_center - pred_center[:, :1]
    aligned = pred_anchor.clone()
    raw_scales = []
    positive_scales = []

    for batch_idx in range(pred_center.shape[0]):
        pred = pred_anchor[batch_idx, 1:]
        gt = gt_center[batch_idx, 1:]
        denominator = pred.square().sum().clamp(min=1e-8)
        raw_scale = (pred * gt).sum() / denominator
        positive_scale = raw_scale.clamp(min=1e-6, max=1e6)
        aligned[batch_idx] = pred_anchor[batch_idx] * positive_scale
        raw_scales.append(raw_scale)
        positive_scales.append(positive_scale)

    raw_scales = torch.stack(raw_scales)
    positive_scales = torch.stack(positive_scales)
    translation_error = torch.norm(aligned[:, 1:] - gt_center[:, 1:], dim=-1)
    return {
        "pose_anchor_raw_scale": float(raw_scales.mean().item()),
        "pose_anchor_positive_scale": float(positive_scales.mean().item()),
        "pose_anchor_negative_scale_fraction": float((raw_scales < 0).float().mean().item()),
        "pose_anchor_trans_rmse_m": tensor_rmse(translation_error),
        "pose_anchor_trans_mean_m": tensor_mean(translation_error),
    }


def compute_pose_metrics_for_convention(pose_enc, pose_gt, convention, sim3_scale):
    center_pred = pose_enc[:, :, :3].float()
    quat_pred = pose_enc[:, :, 3:7].float()
    center_gt = pose_gt[:, :, :3, 3].float()
    rot_gt = pose_gt[:, :, :3, :3].float()
    rot_pred = pose_quaternion_to_rotation(quat_pred, convention)

    batch_size, num_views = center_pred.shape[:2]
    metrics = {}
    prefix = f"pose_{convention}"

    if num_views > 1:
        abs_rot = rotation_angle_rad(rot_pred[:, 1:], rot_gt[:, 1:]) * (180.0 / np.pi)
        anchor_rot_pred = rot_pred[:, :1].transpose(-1, -2) @ rot_pred
        anchor_rot = rotation_angle_rad(anchor_rot_pred[:, 1:], rot_gt[:, 1:]) * (180.0 / np.pi)
        metrics.update({
            f"{prefix}_abs_rot_mean_deg": tensor_mean(abs_rot),
            f"{prefix}_abs_rot_median_deg": tensor_median(abs_rot),
            f"{prefix}_abs_rot_max_deg": tensor_max(abs_rot),
            f"{prefix}_anchor_rot_mean_deg": tensor_mean(anchor_rot),
            f"{prefix}_anchor_rot_median_deg": tensor_median(anchor_rot),
        })
    else:
        metrics.update({
            f"{prefix}_abs_rot_mean_deg": float("nan"),
            f"{prefix}_abs_rot_median_deg": float("nan"),
            f"{prefix}_abs_rot_max_deg": float("nan"),
            f"{prefix}_anchor_rot_mean_deg": float("nan"),
            f"{prefix}_anchor_rot_median_deg": float("nan"),
        })

    if num_views < 2:
        metrics.update({
            f"{prefix}_rpe_rot_mean_deg": float("nan"),
            f"{prefix}_rpe_rot_median_deg": float("nan"),
            f"{prefix}_rpe_trans_rmse_m": float("nan"),
            f"{prefix}_rpe_trans_mean_m": float("nan"),
            f"{prefix}_rpe_trans_dir_mean_deg": float("nan"),
            f"{prefix}_rpe_center_dist_rmse_m": float("nan"),
        })
        for threshold in POSE_AUC_THRESHOLDS:
            metrics.update({
                f"{prefix}_auc{threshold}": float("nan"),
                f"{prefix}_racc{threshold}": float("nan"),
                f"{prefix}_tacc{threshold}": float("nan"),
            })
        return metrics

    pair_i, pair_j = torch.triu_indices(num_views, num_views, offset=1, device=center_pred.device)
    rot_i = rot_pred[:, pair_i]
    rot_j = rot_pred[:, pair_j]
    rot_gt_i = rot_gt[:, pair_i]
    rot_gt_j = rot_gt[:, pair_j]
    rel_rot_pred = rot_i.transpose(-1, -2) @ rot_j
    rel_rot_gt = rot_gt_i.transpose(-1, -2) @ rot_gt_j
    rel_rot_deg = rotation_angle_rad(rel_rot_pred, rel_rot_gt) * (180.0 / np.pi)

    center_i = center_pred[:, pair_i]
    center_j = center_pred[:, pair_j]
    center_gt_i = center_gt[:, pair_i]
    center_gt_j = center_gt[:, pair_j]
    scale = sim3_scale.view(batch_size, 1, 1).to(center_pred.device)
    rel_trans_pred = scale * (rot_i.transpose(-1, -2) @ (center_j - center_i).unsqueeze(-1)).squeeze(-1)
    rel_trans_gt = (rot_gt_i.transpose(-1, -2) @ (center_gt_j - center_gt_i).unsqueeze(-1)).squeeze(-1)
    rel_trans_error = torch.norm(rel_trans_pred - rel_trans_gt, dim=-1)

    pred_pair_distance = scale.squeeze(-1) * torch.norm(center_j - center_i, dim=-1)
    gt_pair_distance = torch.norm(center_gt_j - center_gt_i, dim=-1)
    pair_distance_error = pred_pair_distance - gt_pair_distance

    pred_norm = torch.norm(rel_trans_pred, dim=-1)
    gt_norm = torch.norm(rel_trans_gt, dim=-1)
    dir_valid = (pred_norm > 1e-8) & (gt_norm > 1e-8)
    if dir_valid.any():
        dir_cos = (
            (rel_trans_pred * rel_trans_gt).sum(dim=-1)[dir_valid]
            / (pred_norm[dir_valid] * gt_norm[dir_valid]).clamp(min=1e-8)
        ).clamp(min=-1.0, max=1.0)
        dir_deg = torch.acos(dir_cos) * (180.0 / np.pi)
        dir_mean = tensor_mean(dir_deg)
    else:
        dir_mean = float("nan")

    metrics.update({
        f"{prefix}_rpe_rot_mean_deg": tensor_mean(rel_rot_deg),
        f"{prefix}_rpe_rot_median_deg": tensor_median(rel_rot_deg),
        f"{prefix}_rpe_trans_rmse_m": tensor_rmse(rel_trans_error),
        f"{prefix}_rpe_trans_mean_m": tensor_mean(rel_trans_error),
        f"{prefix}_rpe_trans_dir_mean_deg": dir_mean,
        f"{prefix}_rpe_center_dist_rmse_m": tensor_rmse(pair_distance_error),
    })

    # LingBot benchmark-style pairwise relative-pose AUC: C2W poses are
    # converted to W2C, aligned to the first camera, then evaluated by angular
    # rotation and translation-direction errors over all unordered pairs.
    c2w_pred = build_pose_matrix(rot_pred, center_pred)
    c2w_gt = build_pose_matrix(rot_gt, center_gt)
    pred_extrinsics = align_to_first_camera(invert_transform(c2w_pred))
    gt_extrinsics = align_to_first_camera(invert_transform(c2w_gt))
    rel_pose_pred = invert_transform(pred_extrinsics[:, pair_i]) @ pred_extrinsics[:, pair_j]
    rel_pose_gt = invert_transform(gt_extrinsics[:, pair_i]) @ gt_extrinsics[:, pair_j]
    auc_rot_deg = rotation_angle_rad(rel_pose_pred[..., :3, :3], rel_pose_gt[..., :3, :3]) * (180.0 / np.pi)
    auc_trans_deg = translation_angle_deg(rel_pose_pred[..., :3, 3], rel_pose_gt[..., :3, 3])
    auc_metrics = calculate_pose_auc(auc_rot_deg.reshape(-1), auc_trans_deg.reshape(-1))
    metrics.update({
        f"{prefix}_{metric_name}": metric_value
        for metric_name, metric_value in auc_metrics.items()
    })
    return metrics


def compute_pose_metrics(pose_enc, pose_gt):
    center_pred = pose_enc[:, :, :3].float()
    center_gt = pose_gt[:, :, :3, 3].float()
    sim3_center, sim3_scale, _ = sim3_align_centers(center_pred, center_gt)
    ate_error = torch.norm(sim3_center - center_gt, dim=-1)

    metrics = {
        "pose_ate_sim3_rmse_m": tensor_rmse(ate_error),
        "pose_ate_sim3_mean_m": tensor_mean(ate_error),
        "pose_ate_sim3_max_m": tensor_max(ate_error),
        "pose_sim3_scale": float(sim3_scale.mean().item()),
        "pose_pred_center_norm_mean_m": tensor_mean(torch.norm(center_pred - center_pred[:, :1], dim=-1)[:, 1:]),
        "pose_gt_center_norm_mean_m": tensor_mean(torch.norm(center_gt, dim=-1)[:, 1:]),
    }
    metrics.update(compute_scale_anchor_info(pose_enc, pose_gt))
    metrics.update(compute_pose_metrics_for_convention(pose_enc, pose_gt, "xyzw", sim3_scale))
    metrics.update(compute_pose_metrics_for_convention(pose_enc, pose_gt, "wxyz", sim3_scale))
    return metrics


def align_depth_to_gt(depth_pred, depth_gt, valid_mask_gt):
    if depth_pred.dim() == 5:
        depth_pred = depth_pred.squeeze(-1)
    batch_size, num_views, pred_h, pred_w = depth_pred.shape
    _, _, gt_h, gt_w = depth_gt.shape
    if (pred_h, pred_w) != (gt_h, gt_w):
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.reshape(batch_size * num_views, 1, pred_h, pred_w),
            size=(gt_h, gt_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_views, gt_h, gt_w)
    return depth_pred, valid_mask_gt


def build_lzw_model(device, dinov2_repo, dinov2_hub_name, seed, use_sdpa):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = create_lzw_map_stage1(
        pretrained_path="",
        enable_point=False,
        use_sdpa=use_sdpa,
    )
    load_backbone_from_original_dinov2_repo(model, dinov2_repo, dinov2_hub_name)
    return model.to(device).eval()


def load_trained_lzw_model(checkpoint, device, dinov2_repo, dinov2_hub_name, seed, use_sdpa):
    model = build_lzw_model(device, dinov2_repo, dinov2_hub_name, seed, use_sdpa)
    state = checkpoint.get("trainable_state_dict", checkpoint.get("full_model_state_dict"))
    if state is None:
        raise RuntimeError("LZW checkpoint has no trainable_state_dict or full_model_state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [
        key for key in missing
        if not key.startswith("aggregator.patch_embed.")
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"LZW checkpoint mismatch: missing={relevant_missing[:8]}, unexpected={unexpected[:8]}"
        )
    return model.eval()


def forward_model(model, model_kind, images):
    if model_kind == "lingbot":
        return forward_aggregator_heads(model, images)
    if model_kind == "lzw":
        return forward_lzw_stage1(model, images)
    raise ValueError(f"Unknown model_kind: {model_kind}")


def evaluate_model(model, model_kind, dataset, data_root, device, num_samples, num_vis, label):
    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()
    metrics = {
        "metric_depth_loss": [],
        "scale_aligned_depth_loss": [],
        "legacy_wxyz_scale_aligned_abs_pose_loss": [],
        "legacy_wxyz_scale_aligned_rel_pose_loss": [],
        "scale_aligned_abs_pose_loss": [],
        "scale_aligned_rel_pose_loss": [],
    }
    visuals = {
        "metric": [],
        "aligned": [],
        "pose_enc": [],
        "pose_frame_ids": [],
        "pose_metrics": [],
    }

    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        frame_ids = sample["frame_ids"]
        image_paths = [
            str(Path(data_root) / "results" / f"frame{frame_id:06d}.jpg")
            for frame_id in frame_ids
        ]
        images = load_and_preprocess_images(
            image_paths,
            mode="crop",
            image_size=518,
            patch_size=14,
        ).unsqueeze(0).to(device)
        depths = sample["depths"].unsqueeze(0).to(device)
        masks = sample["valid_masks"].unsqueeze(0).to(device)
        poses = sample["poses"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast_context(device):
            predictions = forward_model(model, model_kind, images)

        depth_pred, masks = align_depth_to_gt(predictions["depth"], depths, masks)
        depth_pred = depth_pred.float().clamp(min=1e-6)
        depths = depths.float()
        aligned_depth = median_align_depth(depth_pred, depths, masks)
        aligned_pose = scale_align_pose_translation(predictions["pose_enc"].float(), poses.float())

        metric_depth_loss = depth_loss_fn(depth_pred, depths, masks)
        aligned_depth_loss = depth_loss_fn(aligned_depth, depths, masks)
        abs_pose_loss = pose_loss_fn(aligned_pose, poses.float())
        rel_pose_loss = rel_pose_loss_fn(aligned_pose, poses.float())
        pose_metrics = compute_pose_metrics(predictions["pose_enc"].float(), poses.float())

        metrics["metric_depth_loss"].append(float(metric_depth_loss))
        metrics["scale_aligned_depth_loss"].append(float(aligned_depth_loss))
        metrics["legacy_wxyz_scale_aligned_abs_pose_loss"].append(float(abs_pose_loss))
        metrics["legacy_wxyz_scale_aligned_rel_pose_loss"].append(float(rel_pose_loss))
        metrics["scale_aligned_abs_pose_loss"].append(float(abs_pose_loss))
        metrics["scale_aligned_rel_pose_loss"].append(float(rel_pose_loss))
        for metric_name, metric_value in pose_metrics.items():
            metrics.setdefault(metric_name, []).append(metric_value)

        if sample_idx < num_vis:
            visuals["metric"].append(depth_pred[0, 0].detach().cpu().numpy())
            visuals["aligned"].append(aligned_depth[0, 0].detach().cpu().numpy())
            visuals["pose_enc"].append(predictions["pose_enc"][0].detach().cpu().float().numpy())
            visuals["pose_frame_ids"].append([int(frame_id) for frame_id in frame_ids])
        visuals["pose_metrics"].append({
            "sample_idx": sample_idx,
            "frame_ids": [int(frame_id) for frame_id in frame_ids],
            **pose_metrics,
            "legacy_wxyz_scale_aligned_abs_pose_loss": float(abs_pose_loss),
            "legacy_wxyz_scale_aligned_rel_pose_loss": float(rel_pose_loss),
        })

        print(
            f"  [{label} {sample_idx + 1}/{min(num_samples, len(dataset))}] "
            f"depth={float(metric_depth_loss):.4f} "
            f"depth_aligned={float(aligned_depth_loss):.4f} "
            f"auc3={pose_metrics['pose_xyzw_auc3']:.2f} "
            f"auc30={pose_metrics['pose_xyzw_auc30']:.2f} "
            f"ate={pose_metrics['pose_ate_sim3_rmse_m']:.4f}m "
            f"xyzw_rpeR={pose_metrics['pose_xyzw_rpe_rot_mean_deg']:.2f}deg "
            f"xyzw_rpeT={pose_metrics['pose_xyzw_rpe_trans_rmse_m']:.4f}m "
            f"legacy_abs={float(abs_pose_loss):.4f}"
        )

    averaged = {key: float(np.mean(values)) for key, values in metrics.items()}
    return averaged, visuals


def collect_gt_visuals(dataset, num_vis):
    visuals = []
    frame_ids = []
    for idx in range(min(num_vis, len(dataset))):
        sample = dataset[idx]
        visuals.append(sample["depths"][0].numpy())
        frame_ids.append(int(sample["frame_ids"][0]))
    return visuals, frame_ids


def plot_depth_comparison(predictions, gt_visuals, frame_ids, output_path, scale_key):
    model_names = ["LingBot Original", "LZW Init", "LZW Trained", "Ground Truth"]
    num_rows = len(gt_visuals)
    fig, axes = plt.subplots(
        num_rows,
        len(model_names),
        figsize=(20, 4.2 * num_rows),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        valid_gt = gt[np.isfinite(gt) & (gt > 0)]
        vmin = float(np.percentile(valid_gt, 2)) if valid_gt.size else 0.0
        vmax = float(np.percentile(valid_gt, 98)) if valid_gt.size else 1.0
        row_images = [
            predictions["LingBot Original"][scale_key][row],
            predictions["LZW Init"][scale_key][row],
            predictions["LZW Trained"][scale_key][row],
            gt,
        ]
        for col, (name, depth) in enumerate(zip(model_names, row_images)):
            if depth.shape != gt.shape:
                depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            image = axes[row, col].imshow(depth, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(
                f"{name} | frame {frame_ids[row]}\nGT scale {vmin:.2f}-{vmax:.2f} m"
            )
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    scale_title = "Metric Scale" if scale_key == "metric" else "Median-Aligned Scale"
    fig.suptitle(f"LZW-Map Stage1 Depth Comparison ({scale_title})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results, output_path):
    metrics = [
        "metric_depth_loss",
        "scale_aligned_depth_loss",
        "scale_aligned_abs_pose_loss",
        "scale_aligned_rel_pose_loss",
    ]
    titles = [
        "Metric Depth Loss",
        "Scale-Aligned Depth Loss",
        "Scale-Aligned Absolute Pose Loss",
        "Scale-Aligned Relative Pose Loss",
    ]
    names = list(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["#3478bf", "#d68c2f", "#3f995b"]
    for axis, metric, title in zip(axes.flat, metrics, titles):
        values = [results[name][metric] for name in names]
        bars = axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pose_metrics(results, output_path):
    metrics = [
        "pose_xyzw_auc3",
        "pose_xyzw_auc30",
        "pose_ate_sim3_rmse_m",
        "pose_xyzw_rpe_trans_rmse_m",
        "pose_xyzw_anchor_rot_mean_deg",
        "pose_xyzw_rpe_rot_mean_deg",
        "pose_wxyz_anchor_rot_mean_deg",
        "pose_wxyz_rpe_rot_mean_deg",
    ]
    titles = [
        "Official XYZW AUC@3 (%)",
        "Official XYZW AUC@30 (%)",
        "Sim(3) ATE RMSE (m)",
        "Official XYZW RPE Trans RMSE (m)",
        "Official XYZW Anchored Rot (deg)",
        "Official XYZW RPE Rot (deg)",
        "Legacy WXYZ Anchored Rot (deg)",
        "Legacy WXYZ RPE Rot (deg)",
    ]
    names = list(results)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    colors = ["#3478bf", "#d68c2f", "#3f995b"]
    for axis, metric, title in zip(axes.flat, metrics, titles):
        values = [results[name].get(metric, float("nan")) for name in names]
        bars = axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            label = "nan" if not np.isfinite(value) else f"{value:.4f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                0 if not np.isfinite(value) else bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Pose Metrics: Official XYZW vs Legacy WXYZ Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(predictions, gt_visuals, frame_ids, output_path):
    num_rows = len(gt_visuals)
    column_names = [
        "Ground Truth",
        "LingBot |Error|",
        "LZW Init |Error|",
        "LZW Trained |Error|",
    ]
    fig, axes = plt.subplots(
        num_rows,
        len(column_names),
        figsize=(17, 4.2 * num_rows),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        valid = np.isfinite(gt) & (gt > 0)
        valid_gt = gt[valid]
        depth_vmin = float(np.percentile(valid_gt, 2)) if valid_gt.size else 0.0
        depth_vmax = float(np.percentile(valid_gt, 98)) if valid_gt.size else 1.0

        model_depths = [
            predictions["LingBot Original"]["aligned"][row],
            predictions["LZW Init"]["aligned"][row],
            predictions["LZW Trained"]["aligned"][row],
        ]
        errors = []
        for depth in model_depths:
            if depth.shape != gt.shape:
                depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            error = np.abs(depth - gt)
            error[~valid] = 0
            errors.append(error)
        error_values = np.concatenate([error[valid] for error in errors]) if valid.any() else np.array([])
        error_vmax = float(np.percentile(error_values, 95)) if error_values.size else 1.0

        panels = [gt, *errors]
        for col, (name, panel) in enumerate(zip(column_names, panels)):
            is_error = col > 0
            image = axes[row, col].imshow(
                panel,
                cmap="hot" if is_error else "gray",
                vmin=0 if is_error else depth_vmin,
                vmax=error_vmax if is_error else depth_vmax,
            )
            title = f"{name} | frame {frame_ids[row]}"
            if is_error and valid.any():
                title += f"\nmean={panel[valid].mean():.3f} m"
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.suptitle("Scale-Aligned Depth Absolute Error", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pose_outputs(predictions, depth_visual_frame_ids, output_path):
    payload = {
        "depth_visual_frame_ids": depth_visual_frame_ids,
        "pose_enc_samples": {},
    }
    for name, values in predictions.items():
        samples = []
        for sample_idx, pose_enc in enumerate(values.get("pose_enc", [])):
            pose_frame_ids = values.get("pose_frame_ids", [])
            samples.append({
                "sample_idx": sample_idx,
                "frame_ids": pose_frame_ids[sample_idx] if sample_idx < len(pose_frame_ids) else [],
                "pose_enc": pose_enc.tolist(),
            })
        if samples:
            payload["pose_enc_samples"][name] = samples
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_pose_metrics(predictions, output_path):
    payload = {
        name: values.get("pose_metrics", [])
        for name, values in predictions.items()
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LZW-Map Stage1 outputs")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--original_model", required=True)
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--dinov2_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--dinov2_hub_name", default="dinov2_vitb14_reg")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    seed = int(ckpt_args.get("seed", 42))
    arch = checkpoint.get("architecture", {})
    dinov2_repo = args.dinov2_repo or arch.get("dinov2_repo") or DEFAULT_DINOV2_REPO
    dinov2_hub_name = args.dinov2_hub_name or arch.get("dinov2_hub_name") or "dinov2_vitb14_reg"

    output_dir = Path(args.output_dir or (Path(args.trained_checkpoint).parent / "vis_validation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=args.num_views,
        max_dim=518,
        shuffle_view_order=False,
        sampler_type="temporal_nearby",
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
        seed=seed,
    )
    gt_visuals, frame_ids = collect_gt_visuals(dataset, args.num_vis)

    print("=" * 72)
    print("LZW-Map Stage1 Validation")
    print("=" * 72)
    print(f"  LingBot original: {args.original_model}")
    print(f"  LZW trained:      {args.trained_checkpoint}")
    print(f"  DINOv2 repo:      {dinov2_repo}")
    print(f"  Samples/views:    {args.num_samples}/{args.num_views}")
    print(f"  Output:           {output_dir}")
    if args.num_views < 3:
        print("  Note: Sim(3) ATE is degenerate with fewer than 3 views; use RPE metrics for this run.")
    print("=" * 72)

    results = {}
    predictions = {}

    model = load_depth_gct_model(args.original_model, args.device, args.use_sdpa)
    results["LingBot Original"], predictions["LingBot Original"] = evaluate_model(
        model,
        "lingbot",
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "LingBot",
    )
    del model
    gc.collect()
    maybe_empty_cache()

    model = build_lzw_model(args.device, dinov2_repo, dinov2_hub_name, seed, args.use_sdpa)
    results["LZW Init"], predictions["LZW Init"] = evaluate_model(
        model,
        "lzw",
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "LZWInit",
    )
    del model
    gc.collect()
    maybe_empty_cache()

    model = load_trained_lzw_model(
        checkpoint,
        args.device,
        dinov2_repo,
        dinov2_hub_name,
        seed,
        args.use_sdpa,
    )
    results["LZW Trained"], predictions["LZW Trained"] = evaluate_model(
        model,
        "lzw",
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "LZWTrained",
    )
    del model
    gc.collect()
    maybe_empty_cache()

    payload = {
        "metadata": {
            "original_model": args.original_model,
            "trained_checkpoint": args.trained_checkpoint,
            "iteration": checkpoint.get("iteration"),
            "seed": seed,
            "dinov2_repo": dinov2_repo,
            "dinov2_hub_name": dinov2_hub_name,
            "num_samples": args.num_samples,
            "num_views": args.num_views,
            "depth_alignment": "per-view GT median scale",
            "pose_alignment": {
                "ate": "Umeyama Sim(3) alignment of predicted camera centers to GT centers",
                "rpe_translation": "relative translation scaled by Sim(3) scale",
                "legacy_loss": "first-frame anchor plus one positive translation scale per sample",
                "auc": (
                    "pairwise relative-pose AUC@{3,5,15,30} in percent; C2W poses are converted "
                    "to W2C, aligned to the first camera, then scored by max(rotation angular error, "
                    "translation-direction angular error)"
                ),
            },
            "pose_quaternion_conventions": {
                "xyzw": "official LingBot pose encoding, scalar-last",
                "wxyz": "legacy local training-loss interpretation, scalar-first",
            },
        },
        "results": results,
    }
    with open(output_dir / "lzw_map_validation_results.json", "w") as handle:
        json.dump(payload, handle, indent=2)

    plot_depth_comparison(
        predictions,
        gt_visuals,
        frame_ids,
        output_dir / "depth_comparison_metric.png",
        "metric",
    )
    plot_depth_comparison(
        predictions,
        gt_visuals,
        frame_ids,
        output_dir / "depth_comparison_aligned.png",
        "aligned",
    )
    plot_metrics(results, output_dir / "metric_comparison.png")
    plot_pose_metrics(results, output_dir / "pose_metric_comparison.png")
    plot_error_comparison(predictions, gt_visuals, frame_ids, output_dir / "depth_error_comparison.png")
    save_pose_outputs(predictions, frame_ids, output_dir / "pose_outputs_sample0.json")
    save_pose_metrics(predictions, output_dir / "pose_metrics_per_sample.json")

    print("\n" + "-" * 102)
    print(
        f"{'Model':<20} {'Depth':>10} {'DepthA':>10} "
        f"{'AUC@3':>9} {'AUC@30':>9} {'ATE(m)':>10} "
        f"{'XYZW RPE-R':>12} {'XYZW RPE-T':>12}"
    )
    print("-" * 102)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['metric_depth_loss']:>10.5f} "
            f"{metrics['scale_aligned_depth_loss']:>10.5f} "
            f"{metrics['pose_xyzw_auc3']:>9.2f} "
            f"{metrics['pose_xyzw_auc30']:>9.2f} "
            f"{metrics['pose_ate_sim3_rmse_m']:>10.5f} "
            f"{metrics['pose_xyzw_rpe_rot_mean_deg']:>12.4f} "
            f"{metrics['pose_xyzw_rpe_trans_rmse_m']:>12.5f}"
        )
    print("-" * 102)

    print("\nOfficial XYZW pose metrics (AUC higher is better; errors lower are better)")
    print("-" * 112)
    print(
        f"{'Model':<20} {'AUC@3':>9} {'AUC@5':>9} {'AUC@15':>9} {'AUC@30':>9} "
        f"{'AbsRot':>10} {'AnchRot':>10} {'RPERot':>10} {'RPETrans':>10} {'Sim3Scale':>10}"
    )
    print("-" * 112)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['pose_xyzw_auc3']:>9.2f} "
            f"{metrics['pose_xyzw_auc5']:>9.2f} "
            f"{metrics['pose_xyzw_auc15']:>9.2f} "
            f"{metrics['pose_xyzw_auc30']:>9.2f} "
            f"{metrics['pose_xyzw_abs_rot_mean_deg']:>10.4f} "
            f"{metrics['pose_xyzw_anchor_rot_mean_deg']:>10.4f} "
            f"{metrics['pose_xyzw_rpe_rot_mean_deg']:>10.4f} "
            f"{metrics['pose_xyzw_rpe_trans_rmse_m']:>10.5f} "
            f"{metrics['pose_sim3_scale']:>10.4f}"
        )
    print("-" * 112)

    print("\nLegacy WXYZ diagnostics (matches the old local PoseLoss convention)")
    print("-" * 92)
    print(
        f"{'Model':<20} {'AbsRot':>10} {'AnchRot':>10} "
        f"{'RPERot':>10} {'OldAbsLoss':>12} {'OldRelLoss':>12}"
    )
    print("-" * 92)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['pose_wxyz_abs_rot_mean_deg']:>10.4f} "
            f"{metrics['pose_wxyz_anchor_rot_mean_deg']:>10.4f} "
            f"{metrics['pose_wxyz_rpe_rot_mean_deg']:>10.4f} "
            f"{metrics['legacy_wxyz_scale_aligned_abs_pose_loss']:>12.5f} "
            f"{metrics['legacy_wxyz_scale_aligned_rel_pose_loss']:>12.5f}"
        )
    print("-" * 92)

    initial = results["LZW Init"]
    trained = results["LZW Trained"]
    if initial["metric_depth_loss"] > 0:
        print(
            "LZW trained vs init metric-depth improvement: "
            f"{(initial['metric_depth_loss'] - trained['metric_depth_loss']) / initial['metric_depth_loss'] * 100:+.1f}%"
        )
    if initial["scale_aligned_depth_loss"] > 0:
        print(
            "LZW trained vs init aligned-depth improvement: "
            f"{(initial['scale_aligned_depth_loss'] - trained['scale_aligned_depth_loss']) / initial['scale_aligned_depth_loss'] * 100:+.1f}%"
        )
    print(f"Validation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
