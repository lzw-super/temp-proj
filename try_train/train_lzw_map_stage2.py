"""Train LZW-Map Stage2 with GCA streaming from a Stage1 checkpoint.

This is the LZW-Map counterpart of the existing GCT Stage2 pilot:
  - Initialize from a LZW-Map Stage1 checkpoint.
  - Reload and freeze the original DINOv2 ViT-B patch_embed backbone.
  - Use foldback long-sequence sampling and progressive view curriculum.
  - Use GCA streaming forward: local sliding window k, num_frame_per_block=1,
    and causal CameraCausalHead KV cache.
  - By default, continue training LZW aggregator blocks + camera/depth heads.
    Use --head_only to freeze the whole aggregator and train only heads.

Usage:
  python try_train/train_lzw_map_stage2.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --dinov2_repo /home/lizhengwu/desktop/temp_proj/dinov2 \
    --stage1_checkpoint try_train/checkpoints/lzw_map_stage1_frozen_dinov2_2-20_spatial_nearby_5000iters/checkpoint_final.pt \
    --output_dir try_train/checkpoints/lzw_map_stage2_smoke \
    --total_iterations 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from foldback_video_sampler import (  # noqa: E402
    FoldbackVideoSampler,
    LocalWindowSampler,
    ProgressiveViewCurriculum,
)
from head_only_model import DepthLoss, PoseLoss  # noqa: E402
from train_replica_stage2 import (  # noqa: E402
    LocalRelativePoseLoss,
    ReplicaLongSequenceDataset,
)
from train_lzw_map_stage1 import (  # noqa: E402
    DEFAULT_DINOV2_REPO,
    align_depth_to_gt,
    load_backbone_from_original_dinov2_repo,
)
from lingbot_map.models.lzw_map import (  # noqa: E402
    create_lzw_map_stage2,
    format_parameter_rows,
    lzw_map_parameter_summary,
)


DEPTH_MIN = 1e-4
DEPTH_MAX = 100.0


def build_scheduler(optimizer, total_iterations, warmup_ratio, min_lr, lr):
    warmup_iterations = max(1, int(total_iterations * warmup_ratio))
    if warmup_iterations >= total_iterations:
        return LinearLR(
            optimizer,
            start_factor=max(min_lr / lr, 1e-4),
            end_factor=1.0,
            total_iters=max(1, total_iterations),
        )
    warmup = LinearLR(
        optimizer,
        start_factor=max(min_lr / lr, 1e-4),
        end_factor=1.0,
        total_iters=warmup_iterations,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_iterations - warmup_iterations,
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iterations],
    )


def count_replica_frames(data_root: str) -> int:
    traj_file = Path(data_root) / "traj.txt"
    if not traj_file.exists():
        raise FileNotFoundError(f"Replica traj.txt not found: {traj_file}")
    with traj_file.open("r") as handle:
        return sum(1 for line in handle if len(line.strip().split()) == 16)


def _compute_pointcloud_anchor_scale(
    depths,
    valid_masks,
    poses,
    intrinsics,
    num_anchor,
    sample_stride=4,
):
    """Paper-style anchor scale: mean distance of anchor-frame GT points."""
    batch_size, _, height, width = depths.shape
    stride = max(1, int(sample_stride))
    u = torch.arange(0, width, stride, device=depths.device, dtype=depths.dtype)
    v = torch.arange(0, height, stride, device=depths.device, dtype=depths.dtype)
    yy, xx = torch.meshgrid(v, u, indexing="ij")
    scale = torch.ones(batch_size, device=depths.device, dtype=depths.dtype)

    for batch_idx in range(batch_size):
        distances = []
        for frame_idx in range(num_anchor):
            depth = depths[batch_idx, frame_idx, ::stride, ::stride]
            mask = valid_masks[batch_idx, frame_idx, ::stride, ::stride].bool()
            finite_mask = mask & torch.isfinite(depth) & (depth > DEPTH_MIN)
            if finite_mask.sum() < 10:
                continue

            K = intrinsics[batch_idx, frame_idx].to(device=depths.device, dtype=depths.dtype)
            fx = K[0, 0].clamp(min=1e-6)
            fy = K[1, 1].clamp(min=1e-6)
            cx = K[0, 2]
            cy = K[1, 2]

            z = depth[finite_mask]
            x = (xx[finite_mask] - cx) / fx * z
            y = (yy[finite_mask] - cy) / fy * z
            points_cam = torch.stack([x, y, z], dim=-1)

            pose = poses[batch_idx, frame_idx].to(device=depths.device, dtype=depths.dtype)
            points_world = points_cam @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
            distances.append(points_world.norm(dim=-1))

        if distances:
            valid_distances = torch.cat(distances)
            if valid_distances.numel() >= 10:
                scale[batch_idx] = valid_distances.float().mean().clamp(min=1e-3).to(depths.dtype)

    return scale


def compute_anchor_scale(
    depths,
    valid_masks,
    poses,
    num_anchor_frames,
    source="pointcloud_mean",
    intrinsics=None,
    sample_stride=4,
):
    """Compute per-sample anchor scale for Stage2 depth/translation targets."""
    batch_size, num_views = depths.shape[:2]
    num_anchor = min(num_anchor_frames, num_views)
    scale = torch.ones(batch_size, device=depths.device, dtype=depths.dtype)

    if source == "pointcloud_mean":
        if intrinsics is None:
            raise ValueError("pointcloud_mean anchor scale requires batch intrinsics")
        return _compute_pointcloud_anchor_scale(
            depths,
            valid_masks,
            poses,
            intrinsics,
            num_anchor,
            sample_stride=sample_stride,
        )

    for batch_idx in range(batch_size):
        if source == "depth_median":
            anchor_depth = depths[batch_idx, :num_anchor]
            anchor_mask = valid_masks[batch_idx, :num_anchor].bool()
            valid_depth = anchor_depth[anchor_mask]
            if valid_depth.numel() >= 10:
                scale[batch_idx] = valid_depth.float().median().clamp(min=1e-3)
        elif source == "translation_norm":
            if num_anchor > 1:
                translations = poses[batch_idx, 1:num_anchor, :3, 3]
                scale[batch_idx] = translations.float().norm(dim=-1).mean().clamp(min=1e-3)
        else:
            raise ValueError(f"Unknown anchor scale source: {source}")

    return scale


def apply_anchor_scale_normalization(depths, poses, scale):
    batch_size = depths.shape[0]
    depths_norm = depths / scale.view(batch_size, 1, 1, 1)
    poses_norm = poses.clone()
    poses_norm[:, :, :3, 3] = poses[:, :, :3, 3] / scale.view(batch_size, 1, 1)
    return depths_norm, poses_norm


def trainable_state_dict(model):
    """Save trainable LZW modules, excluding the frozen original DINO patch_embed."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("aggregator.patch_embed.")
    }


def _state_from_checkpoint(checkpoint):
    for key in (
        "trainable_state_dict",
        "full_model_state_dict",
        "model_state_dict",
        "head_state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key], key
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "raw_state_dict"
    raise RuntimeError("Checkpoint has no supported model state dict")


def load_lzw_state(model, checkpoint_path, allow_head_only=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, state_key = _state_from_checkpoint(checkpoint)
    state_dict = dict(state_dict)
    if (
        hasattr(model.aggregator, "anchor_token")
        and not hasattr(model.aggregator, "scale_token")
        and "aggregator.anchor_token" not in state_dict
        and "aggregator.scale_token" in state_dict
    ):
        state_dict["aggregator.anchor_token"] = state_dict.pop("aggregator.scale_token")
        print("[checkpoint] Remapped legacy aggregator.scale_token -> aggregator.anchor_token")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    allowed_missing_prefixes = ["aggregator.patch_embed."]
    if allow_head_only and state_key == "head_state_dict":
        allowed_missing_prefixes.append("aggregator.")

    relevant_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch for {checkpoint_path}: "
            f"missing={relevant_missing[:8]}, unexpected={unexpected[:8]}"
        )

    print(f"[checkpoint] Loaded {state_key} from {checkpoint_path}")
    print(f"[checkpoint] Iteration: {checkpoint.get('iteration', -1)}")
    print(f"[checkpoint] Loss dict: {checkpoint.get('loss_dict', {})}")
    return checkpoint


def configure_trainable_params(model, head_only=False):
    for param in model.parameters():
        param.requires_grad = True

    if head_only:
        for param in model.aggregator.parameters():
            param.requires_grad = False
    else:
        for param in model.aggregator.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(model.aggregator.patch_embed, "mask_token"):
            model.aggregator.patch_embed.mask_token.requires_grad_(False)

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    mode = "head-only" if head_only else "lzw aggregator + heads"
    print(f"[freeze] Train mode: {mode}")
    print(f"[freeze] Trainable params: {trainable:,} ({trainable / total:.2%})")
    print(f"[freeze] Frozen params: {frozen:,} ({frozen / total:.2%})")


def set_training_modes(model, head_only=False):
    model.train()
    if head_only:
        model.aggregator.eval()
    else:
        model.aggregator.train()
    model.aggregator.patch_embed.eval()


def _append_feature_chunks(feature_chunks, features):
    if feature_chunks is None:
        return [[feature] for feature in features]
    if len(feature_chunks) != len(features):
        raise RuntimeError(
            f"Feature tap count changed during streaming: "
            f"{len(feature_chunks)} != {len(features)}"
        )
    for chunks, feature in zip(feature_chunks, features):
        chunks.append(feature)
    return feature_chunks


def _concat_camera_outputs(camera_outputs):
    if not camera_outputs:
        return {}

    result = {}
    if "pose_enc" in camera_outputs[0]:
        result["pose_enc"] = torch.cat(
            [output["pose_enc"] for output in camera_outputs],
            dim=1,
        )
    if "pose_enc_list" in camera_outputs[0]:
        num_iterations = len(camera_outputs[0]["pose_enc_list"])
        result["pose_enc_list"] = [
            torch.cat(
                [output["pose_enc_list"][idx] for output in camera_outputs],
                dim=1,
            )
            for idx in range(num_iterations)
        ]
    return result


def forward_lzw_stage2_streaming(
    model,
    images,
    sliding_window_size,
    num_frame_for_scale=8,
    head_only=False,
):
    """Stage2 GCA streaming forward for LZW-Map."""
    _, num_views = images.shape[:2]
    scale_frames = min(num_frame_for_scale, num_views)

    model.clean_kv_cache()
    feature_chunks = None
    camera_outputs = []
    patch_start_idx = None

    def _aggregate_chunk(chunk_images, frames_per_block):
        aggregate_context = torch.no_grad() if head_only else contextlib.nullcontext()
        with aggregate_context:
            chunk_features, chunk_patch_start_idx = model.aggregator(
                chunk_images,
                selected_idx=model.selected_idx,
                num_frame_for_scale=scale_frames,
                sliding_window_size=sliding_window_size,
                num_frame_per_block=frames_per_block,
            )
        if head_only:
            chunk_features = [feature.detach() for feature in chunk_features]
        return chunk_features, chunk_patch_start_idx

    scale_images = images[:, :scale_frames]
    scale_features, patch_start_idx = _aggregate_chunk(
        scale_images,
        frames_per_block=scale_frames,
    )
    feature_chunks = _append_feature_chunks(feature_chunks, scale_features)
    camera_outputs.append(
        model._predict_camera(
            scale_features,
            causal_inference=True,
            num_frame_per_block=scale_frames,
            num_frame_for_scale=scale_frames,
            sliding_window_size=sliding_window_size,
        )
    )

    for frame_idx in range(scale_frames, num_views):
        frame_features, frame_patch_start_idx = _aggregate_chunk(
            images[:, frame_idx:frame_idx + 1],
            frames_per_block=1,
        )
        if frame_patch_start_idx != patch_start_idx:
            raise RuntimeError(
                f"Patch start index changed during streaming: "
                f"{frame_patch_start_idx} != {patch_start_idx}"
            )
        feature_chunks = _append_feature_chunks(feature_chunks, frame_features)
        camera_outputs.append(
            model._predict_camera(
                frame_features,
                causal_inference=True,
                num_frame_per_block=1,
                num_frame_for_scale=scale_frames,
                sliding_window_size=sliding_window_size,
            )
        )

    features = [torch.cat(chunks, dim=1) for chunks in feature_chunks]
    camera_output = _concat_camera_outputs(camera_outputs)

    model.clean_kv_cache()

    depth_output = model._predict_depth(
        features,
        images=images,
        patch_start_idx=patch_start_idx,
    )

    result = {}
    result.update(camera_output)
    result.update(depth_output)
    result["images"] = images
    return result


def _first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return name, int(torch.isnan(grad).sum().item()), int(torch.isinf(grad).sum().item())
    return None


def _assert_trainable_params_finite(model, context):
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite trainable parameter detected after {context}: {name}")


def train_one_iteration(
    model,
    batch,
    optimizer,
    scheduler,
    scaler,
    depth_loss_fn,
    pose_loss_fn,
    rel_pose_loss_fn,
    window_pairs,
    sliding_window_size,
    args,
):
    set_training_modes(model, head_only=args.head_only)

    images = batch["images"].to(args.device, non_blocking=True)
    depths = batch["depths"].to(args.device, non_blocking=True)
    valid_masks = batch["valid_masks"].to(args.device, non_blocking=True)
    poses = batch["poses"].to(args.device, non_blocking=True)
    intrinsics = None
    if "intrinsics" in batch:
        intrinsics = batch["intrinsics"].to(args.device, non_blocking=True)

    anchor_scale = None
    if args.use_anchor_scale_norm:
        anchor_scale = compute_anchor_scale(
            depths,
            valid_masks,
            poses,
            num_anchor_frames=args.num_frame_for_scale,
            source=args.anchor_scale_source,
            intrinsics=intrinsics,
            sample_stride=args.anchor_scale_sample_stride,
        )
        depths, poses = apply_anchor_scale_normalization(depths, poses, anchor_scale)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
        predictions = forward_lzw_stage2_streaming(
            model,
            images,
            sliding_window_size=sliding_window_size,
            num_frame_for_scale=args.num_frame_for_scale,
            head_only=args.head_only,
        )

        depth_pred, valid_masks_aligned = align_depth_to_gt(
            predictions["depth"],
            depths,
            valid_masks,
        )
        if not torch.isfinite(depth_pred).all():
            optimizer.zero_grad(set_to_none=True)
            return {"total": float("nan"), "depth": float("nan"), "skipped": 1.0}

        depth_pred = depth_pred.float().clamp(min=DEPTH_MIN, max=DEPTH_MAX)
        depths_safe = torch.nan_to_num(
            depths.float(),
            nan=1.0,
            posinf=DEPTH_MAX,
            neginf=DEPTH_MIN,
        ).clamp(min=DEPTH_MIN, max=DEPTH_MAX)

        depth_loss = depth_loss_fn(depth_pred, depths_safe, valid_masks_aligned)
        total_loss = depth_loss
        loss_dict = {
            "depth": float(depth_loss.detach()),
            "depth_min": float(depth_pred.detach().amin()),
            "depth_max": float(depth_pred.detach().amax()),
        }
        if anchor_scale is not None:
            loss_dict["anchor_s"] = float(anchor_scale.detach().mean())

        pose_enc = predictions.get("pose_enc")
        if pose_enc is not None:
            if not torch.isfinite(pose_enc).all():
                optimizer.zero_grad(set_to_none=True)
                return {"total": float("nan"), "depth": loss_dict["depth"], "skipped": 1.0}
            pose_enc = pose_enc.float()

        if pose_loss_fn is not None and pose_enc is not None and args.pose_weight != 0:
            abs_pose_loss = pose_loss_fn(pose_enc, poses.float())
            total_loss = total_loss + args.pose_weight * abs_pose_loss
            loss_dict["abs_pose"] = float(abs_pose_loss.detach())
        else:
            loss_dict["abs_pose"] = 0.0

        if (
            rel_pose_loss_fn is not None
            and pose_enc is not None
            and args.rel_pose_weight != 0
            and args.iteration >= args.rel_pose_start_iter
            and window_pairs
        ):
            rel_pose_loss = rel_pose_loss_fn(pose_enc, poses.float(), window_pairs)
            total_loss = total_loss + args.rel_pose_weight * rel_pose_loss
            loss_dict["rel_pose"] = (
                float(rel_pose_loss.detach())
                if torch.is_tensor(rel_pose_loss)
                else float(rel_pose_loss)
            )
        else:
            loss_dict["rel_pose"] = 0.0

    if not torch.isfinite(total_loss):
        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"] = float("nan")
        loss_dict["skipped"] = 1.0
        return loss_dict

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    skipped_step = False

    if args.use_amp:
        previous_scale = scaler.get_scale()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            args.gradient_clip_norm,
        )
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        else:
            skipped_step = True
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None:
                print(f"[nonfinite_grad] {bad_grad[0]} nan={bad_grad[1]} inf={bad_grad[2]}")
            optimizer.zero_grad(set_to_none=True)
        scaler.update()
        skipped_step = skipped_step or scaler.get_scale() < previous_scale
    else:
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            args.gradient_clip_norm,
        )
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            skipped_step = True
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None:
                print(f"[nonfinite_grad] {bad_grad[0]} nan={bad_grad[1]} inf={bad_grad[2]}")
            optimizer.zero_grad(set_to_none=True)

    _assert_trainable_params_finite(model, f"iteration {args.iteration}")

    if scheduler is not None and not skipped_step:
        scheduler.step()

    loss_dict["total"] = float(total_loss.detach())
    loss_dict["grad_norm"] = float(grad_norm.detach()) if torch.is_tensor(grad_norm) else float(grad_norm)
    loss_dict["skipped"] = 1.0 if skipped_step else 0.0
    return loss_dict


def save_checkpoint(model, optimizer, scheduler, scaler, iteration, loss_dict, args, path):
    checkpoint = {
        "format": "lzw_map_stage2_frozen_dinov2_backbone",
        "model_name": "lzw-map-stage2",
        "iteration": iteration,
        "trainable_state_dict": trainable_state_dict(model),
        "loss_dict": loss_dict,
        "architecture": {
            "patch_embed": "dinov2_vitb14_reg",
            "embed_dim": 768,
            "aggregator_depth": 12,
            "selected_idx": list(model.selected_idx),
            "camera_trunk_depth": model.camera_trunk_depth,
            "camera_num_heads": model.camera_num_heads,
            "enable_point": False,
            "frozen_backbone": True,
            "head_only": args.head_only,
            "strict_gca_streaming": True,
            "use_anchor_token": getattr(model.aggregator, "use_anchor_token", False),
            "use_scale_token": getattr(model.aggregator, "use_scale_token", False),
            "num_special_tokens": getattr(model.aggregator, "num_special_tokens", None),
            "enable_camera_sliding_window": model.enable_camera_sliding_window,
            "max_frame_num": args.max_frame_num,
            "anchor_scale_source": args.anchor_scale_source,
            "dinov2_repo": args.dinov2_repo,
            "dinov2_hub_name": args.dinov2_hub_name,
        },
        "args": vars(args),
    }
    if args.save_full_model_state:
        checkpoint["full_model_state_dict"] = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    if args.save_optimizer_state:
        checkpoint.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        })
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def plot_loss_history(history, output_dir):
    if not history["iteration"]:
        return
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    curves = [
        ("total", "Total Loss"),
        ("depth", "Depth Loss"),
        ("abs_pose", "Absolute Pose"),
        ("rel_pose", "Local Rel Pose"),
        ("grad_norm", "Gradient Norm"),
        ("lr", "Learning Rate"),
        ("views", "Views"),
        ("k", "GCA Window k"),
    ]
    for axis, (key, title) in zip(axes.flat, curves):
        axis.plot(history["iteration"], history[key], linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="LZW-Map Stage2 GCA streaming training")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dinov2_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--dinov2_hub_name", default="dinov2_vitb14_reg")
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--max_frame_num", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    parser.add_argument("--pose_weight", type=float, default=0.1)
    parser.add_argument("--rel_pose_weight", type=float, default=0.05)
    parser.add_argument("--rel_pose_start_iter", type=int, default=500)

    parser.add_argument("--views_start", type=int, default=4)
    parser.add_argument("--views_end", type=int, default=8)
    parser.add_argument("--view_variance", type=int, default=2)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--num_frame_for_scale", type=int, default=8)

    parser.add_argument("--use_anchor_scale_norm", action="store_true", default=True)
    parser.add_argument("--no_anchor_scale_norm", dest="use_anchor_scale_norm", action="store_false")
    parser.add_argument(
        "--anchor_scale_source",
        choices=["pointcloud_mean", "depth_median", "translation_norm"],
        default="pointcloud_mean",
    )
    parser.add_argument("--anchor_scale_sample_stride", type=int, default=4)

    parser.add_argument("--stride_min", type=int, default=1)
    parser.add_argument("--stride_max", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-8)

    parser.add_argument("--head_only", action="store_true")
    parser.add_argument("--no_color_aug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--save_full_model_state", action="store_true")
    parser.add_argument("--skip_final_save", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    args = parser.parse_args()

    if args.views_start < 2:
        parser.error("--views_start must be >= 2")
    if args.views_end < args.views_start:
        parser.error("--views_end must be >= --views_start")
    if args.k_min < 1:
        parser.error("--k_min must be >= 1")
    if args.k_max < args.k_min:
        parser.error("--k_max must be >= --k_min")
    if args.max_frame_num < args.views_end:
        parser.error("--max_frame_num must be >= --views_end")

    args.use_amp = not args.no_amp
    if not args.device.startswith("cuda"):
        args.use_amp = False
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("LZW-Map Stage2 Training (GCA streaming)")
    print("=" * 72)
    print(f"  Stage1 checkpoint:  {args.stage1_checkpoint}")
    print(f"  DINOv2 repo:        {args.dinov2_repo}")
    print(f"  DINOv2 hub name:    {args.dinov2_hub_name}")
    print(f"  Train mode:         {'head-only' if args.head_only else 'aggregator blocks + heads'}")
    print(f"  Frozen module:      aggregator.patch_embed")
    print(f"  Views curriculum:   {args.views_start}->{args.views_end}")
    print(f"  Local window k:     [{args.k_min}, {args.k_max}]")
    print(f"  Anchor frames:      {args.num_frame_for_scale}")
    print(f"  LR / WD:            {args.lr:g} / {args.weight_decay:g}")
    print(
        f"  Anchor norm:        {args.use_anchor_scale_norm} "
        f"({args.anchor_scale_source}, stride={args.anchor_scale_sample_stride})"
    )
    print(f"  AMP / SDPA:         {args.use_amp} / {args.use_sdpa}")
    print(f"  Output:             {output_dir}")
    if args.resume:
        print(f"  Resume:             {args.resume}")
    print("=" * 72)

    model = create_lzw_map_stage2(
        pretrained_path="",
        enable_point=False,
        use_sdpa=args.use_sdpa,
        max_frame_num=args.max_frame_num,
    )
    load_backbone_from_original_dinov2_repo(
        model,
        args.dinov2_repo,
        args.dinov2_hub_name,
    )

    init_checkpoint = args.resume or args.stage1_checkpoint
    init_ckpt = load_lzw_state(
        model,
        init_checkpoint,
        allow_head_only=args.head_only,
    )
    print(
        "[model] Context tokens: "
        f"anchor={getattr(model.aggregator, 'use_anchor_token', False)}, "
        f"scale={getattr(model.aggregator, 'use_scale_token', False)}, "
        f"num_special={getattr(model.aggregator, 'num_special_tokens', None)}"
    )
    configure_trainable_params(model, head_only=args.head_only)
    model = model.to(args.device)

    print("[model] Parameter summary")
    for line in format_parameter_rows(lzw_map_parameter_summary(model)):
        print(line)

    if args.dry_run:
        print("[dry_run] Model constructed and checkpoint loaded. Exiting.")
        return

    total_frames = count_replica_frames(args.data_root)
    foldback_sampler = FoldbackVideoSampler(
        total_frames=total_frames,
        stride_range=(args.stride_min, args.stride_max),
        redraw_stride_after_reverse=True,
        seed=args.seed,
    )
    view_curriculum = ProgressiveViewCurriculum(
        views_start=args.views_start,
        views_end=args.views_end,
        total_iterations=args.total_iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
    )
    window_sampler = LocalWindowSampler(
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )
    dataset = ReplicaLongSequenceDataset(
        data_root=args.data_root,
        foldback_sampler=foldback_sampler,
        max_dim=args.img_size,
        color_jitter_prob=0.0 if args.no_color_aug else 0.9,
        seed=args.seed,
    )
    if args.num_workers != 0:
        print("[data] WARNING: --num_workers > 0 may make per-iteration view curriculum stale.")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        args.total_iterations,
        args.warmup_ratio,
        args.min_lr,
        args.lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    start_iteration = 0
    if args.resume:
        if "optimizer_state_dict" in init_ckpt:
            optimizer.load_state_dict(init_ckpt["optimizer_state_dict"])
            if init_ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(init_ckpt["scheduler_state_dict"])
            if init_ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(init_ckpt["scaler_state_dict"])
            print("[resume] Loaded optimizer/scheduler/scaler state")
        else:
            print("[resume] Optimizer state not found; resuming model weights only")
        start_iteration = int(init_ckpt.get("iteration", 0))
        print(f"[resume] Continuing after iteration {start_iteration}")

    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = LocalRelativePoseLoss()

    history = {
        "iteration": [],
        "total": [],
        "depth": [],
        "abs_pose": [],
        "rel_pose": [],
        "grad_norm": [],
        "lr": [],
        "views": [],
        "k": [],
        "skipped": [],
        "anchor_s": [],
        "depth_min": [],
        "depth_max": [],
        "max_allocated_gib": [],
        "max_reserved_gib": [],
    }
    data_iter = iter(dataloader)
    start_time = time.time()
    loss_dict = {"total": 0.0}

    for iteration in range(start_iteration, args.total_iterations):
        current_views = view_curriculum.get_num_views_with_variance(
            iteration,
            variance=args.view_variance,
        )
        current_views = max(2, min(current_views, args.views_end))
        dataset.set_num_views(current_views)

        window_size = min(window_sampler.sample_window_size(), max(1, current_views - 1))
        window_pairs = window_sampler.get_adjacent_pairs(current_views, window_size)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        args.iteration = iteration
        loss_dict = train_one_iteration(
            model,
            batch,
            optimizer,
            scheduler,
            scaler,
            depth_loss_fn,
            pose_loss_fn,
            rel_pose_loss_fn,
            window_pairs,
            window_size,
            args,
        )
        completed = iteration + 1
        current_lr = optimizer.param_groups[0]["lr"]
        max_allocated = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        max_reserved = (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )

        history["iteration"].append(completed)
        history["total"].append(loss_dict.get("total", float("nan")))
        history["depth"].append(loss_dict.get("depth", 0.0))
        history["abs_pose"].append(loss_dict.get("abs_pose", 0.0))
        history["rel_pose"].append(loss_dict.get("rel_pose", 0.0))
        history["grad_norm"].append(loss_dict.get("grad_norm", 0.0))
        history["lr"].append(current_lr)
        history["views"].append(current_views)
        history["k"].append(window_size)
        history["skipped"].append(loss_dict.get("skipped", 0.0))
        history["anchor_s"].append(loss_dict.get("anchor_s", 0.0))
        history["depth_min"].append(loss_dict.get("depth_min", 0.0))
        history["depth_max"].append(loss_dict.get("depth_max", 0.0))
        history["max_allocated_gib"].append(max_allocated)
        history["max_reserved_gib"].append(max_reserved)

        if completed == 1 or completed % args.log_every == 0:
            elapsed = time.time() - start_time
            anchor_text = (
                f" anchor_s={loss_dict.get('anchor_s', 0.0):.3f}"
                if "anchor_s" in loss_dict
                else ""
            )
            print(
                f"[Iter {completed}/{args.total_iterations}] "
                f"views={current_views} k={window_size} "
                f"total={loss_dict.get('total', float('nan')):.5f} "
                f"depth={loss_dict.get('depth', 0.0):.5f} "
                f"abs_pose={loss_dict.get('abs_pose', 0.0):.5f} "
                f"rel_pose={loss_dict.get('rel_pose', 0.0):.5f}"
                f"{anchor_text} "
                f"grad={loss_dict.get('grad_norm', 0.0):.4f} "
                f"skip={int(loss_dict.get('skipped', 0.0))} "
                f"lr={current_lr:.2e} "
                f"max_mem={max_allocated:.2f}/{max_reserved:.2f} GiB "
                f"time={elapsed:.1f}s"
            )

        if args.save_every > 0 and completed % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                completed,
                loss_dict,
                args,
                output_dir / f"checkpoint_iter_{completed}.pt",
            )

        if torch.cuda.is_available() and completed % 100 == 0:
            torch.cuda.empty_cache()

    if args.skip_final_save:
        print("[checkpoint] Skipped final checkpoint (--skip_final_save)")
    else:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            args.total_iterations,
            loss_dict,
            args,
            output_dir / "checkpoint_final.pt",
        )

    with (vis_dir / "loss_history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    plot_loss_history(history, vis_dir)

    print("=" * 72)
    print("LZW-Map Stage2 training complete")
    print(f"  Final total:      {loss_dict.get('total', float('nan')):.6f}")
    print(f"  Final depth:      {loss_dict.get('depth', float('nan')):.6f}")
    print(f"  Final abs pose:   {loss_dict.get('abs_pose', float('nan')):.6f}")
    print(f"  Final rel pose:   {loss_dict.get('rel_pose', float('nan')):.6f}")
    if history["max_allocated_gib"]:
        print(f"  Peak allocated:   {max(history['max_allocated_gib']):.2f} GiB")
        print(f"  Peak reserved:    {max(history['max_reserved_gib']):.2f} GiB")
    print(f"  Output:           {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
