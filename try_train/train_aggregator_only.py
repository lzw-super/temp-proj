"""
Train a configurable, randomly initialized slice of the GCT aggregator.

The DINOv2 patch embed, depth head, and camera head keep their original
LingBot-Map weights and stay frozen. Selected frame/global blocks are optimized
with the Stage1 fused objective: depth + absolute pose + relative pose.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import trunc_normal_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from replica_dataset import create_replica_dataloader
from lingbot_map.layers.layer_scale import LayerScale
from lingbot_map.models.gct_stream import GCTStream


def load_depth_gct_model(checkpoint_path, device, use_sdpa=True):
    """Load the original aggregator, depth head, and camera head."""
    model = GCTStream(
        img_size=518,
        patch_size=14,
        pretrained_path=None,
        enable_camera=True,
        enable_point=False,
        enable_local_point=False,
        enable_depth=True,
        enable_3d_rope=True,
        max_frame_num=100,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        use_gradient_checkpoint=True,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("full_model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict
    gc.collect()

    relevant_missing = [
        key for key in missing
        if key.startswith(("aggregator.", "depth_head.", "camera_head."))
    ]
    if relevant_missing:
        raise RuntimeError(
            f"Missing required aggregator/head weights: {relevant_missing[:8]}"
        )
    if unexpected:
        print(f"[load_model] Ignored checkpoint keys: {len(unexpected)}")

    model = model.to(device)
    print(f"[load_model] Loaded original aggregator/depth/camera weights from {checkpoint_path}")
    return model


def parse_block_indices(spec, depth):
    """Parse expressions such as '12-17', '4,11,17', or '4-6,11,17-20'."""
    if not spec:
        return []
    if spec.strip().lower() == "all":
        return list(range(depth))

    indices = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid block range: {item}")
            start, end = (int(value.strip()) for value in parts)
            if end < start:
                raise ValueError(f"Block range must be ascending: {item}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))

    parsed = sorted(indices)
    invalid = [idx for idx in parsed if not 0 <= idx < depth]
    if invalid:
        raise ValueError(f"Block indices must be in [0, {depth - 1}], got {invalid}")
    if not parsed:
        raise ValueError("No block indices were selected")
    return parsed


def block_indices_to_tag(indices):
    """Convert sorted indices to a compact tag such as '12-17_23'."""
    ranges = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = idx
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return "_".join(ranges)


def selected_block_indices(model, block_indices=None, train_last_n_blocks=None):
    depth = len(model.aggregator.frame_blocks)
    if train_last_n_blocks is not None:
        if not 1 <= train_last_n_blocks <= depth:
            raise ValueError(
                f"train_last_n_blocks must be in [1, {depth}], got {train_last_n_blocks}"
            )
        return list(range(depth - train_last_n_blocks, depth))
    if block_indices:
        return parse_block_indices(block_indices, depth)
    return [17]


def _reset_transformer_module(module):
    """Reset a block with stable ViT-style random initialization."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    elif isinstance(module, LayerScale):
        nn.init.constant_(module.gamma, 0.01)


def random_init_selected_blocks(
    model,
    block_indices,
    train_frame_blocks=True,
    train_global_blocks=True,
    seed=42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    initialized = []
    for idx in block_indices:
        if train_frame_blocks:
            model.aggregator.frame_blocks[idx].apply(_reset_transformer_module)
            initialized.append(f"frame_blocks.{idx}")
        if train_global_blocks:
            model.aggregator.global_blocks[idx].apply(_reset_transformer_module)
            initialized.append(f"global_blocks.{idx}")

    print(f"[random_init] Randomly initialized: {', '.join(initialized)}")


def configure_trainable_blocks(
    model,
    block_indices,
    train_frame_blocks=True,
    train_global_blocks=True,
):
    for param in model.parameters():
        param.requires_grad = False

    for idx in block_indices:
        if train_frame_blocks:
            for param in model.aggregator.frame_blocks[idx].parameters():
                param.requires_grad = True
        if train_global_blocks:
            for param in model.aggregator.global_blocks[idx].parameters():
                param.requires_grad = True

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    if not trainable_names:
        raise RuntimeError("No trainable aggregator parameters were selected")

    print(
        f"[freeze] Trainable aggregator parameters: {trainable_count:,} "
        f"({trainable_count / total_count:.2%} of instantiated model)"
    )
    print(f"[freeze] Trainable tensors: {len(trainable_names)}")
    print(f"[freeze] First trainable tensors: {trainable_names[:8]}")
    return trainable_count


def aggregator_state_dict(model, block_indices, train_frame_blocks, train_global_blocks):
    prefixes = []
    for idx in block_indices:
        if train_frame_blocks:
            prefixes.append(f"aggregator.frame_blocks.{idx}.")
        if train_global_blocks:
            prefixes.append(f"aggregator.global_blocks.{idx}.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(tuple(prefixes))
    }


def forward_aggregator_heads(
    model,
    images,
    predict_depth=True,
    predict_camera=True,
):
    """Stage1 global-attention forward through selected frozen output heads."""
    _, num_views = images.shape[:2]
    model.aggregator.clean_kv_cache()
    features, patch_start_idx = model.aggregator(
        images,
        selected_idx=[4, 11, 17, 23],
        num_frame_for_scale=num_views,
        sliding_window_size=-1,
        num_frame_per_block=num_views,
    )
    model.aggregator.clean_kv_cache()

    result = {}
    if predict_camera:
        model.camera_head.clean_kv_cache()
        camera_output = model._predict_camera(
            features,
            causal_inference=False,
            num_frame_per_block=num_views,
            num_frame_for_scale=num_views,
        )
        model.camera_head.clean_kv_cache()
        result.update(camera_output)
    if predict_depth:
        depth_output = model._predict_depth(
            features,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        result.update(depth_output)
    return result


def forward_aggregator_depth(model, images):
    """Backward-compatible depth-only wrapper used by older callers."""
    return forward_aggregator_heads(
        model,
        images,
        predict_depth=True,
        predict_camera=False,
    )


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


def set_training_modes(model, block_indices, train_frame_blocks, train_global_blocks):
    # Aggregator.train() activates its gradient-checkpoint path.
    model.eval()
    model.aggregator.train()
    model.aggregator.patch_embed.eval()
    model.depth_head.eval()
    model.camera_head.eval()

    for block in model.aggregator.frame_blocks:
        block.eval()
    for block in model.aggregator.global_blocks:
        block.eval()
    for idx in block_indices:
        if train_frame_blocks:
            model.aggregator.frame_blocks[idx].train()
        if train_global_blocks:
            model.aggregator.global_blocks[idx].train()


def _backward(loss, scaler, use_amp):
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def train_one_iteration(
    model,
    batch,
    optimizer,
    scheduler,
    scaler,
    depth_loss_fn,
    pose_loss_fn,
    rel_pose_loss_fn,
    args,
):
    set_training_modes(
        model,
        args.block_indices,
        args.train_frame_blocks,
        args.train_global_blocks,
    )
    images = batch["images"].to(args.device, non_blocking=True)
    depths = batch["depths"].to(args.device, non_blocking=True)
    valid_masks = batch["valid_masks"].to(args.device, non_blocking=True)
    poses = batch["poses"].to(args.device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_rel_pose = args.iteration >= args.rel_pose_start_iter

    if args.split_head_backward:
        # Recompute the aggregator for the pose branch so the two large frozen
        # head graphs are not retained in memory at the same time.
        with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
            depth_predictions = forward_aggregator_heads(
                model,
                images,
                predict_depth=True,
                predict_camera=False,
            )
            depth_pred, depth_masks = align_depth_to_gt(
                depth_predictions["depth"], depths, valid_masks
            )
            depth_loss = depth_loss_fn(depth_pred, depths, depth_masks)
        if not torch.isfinite(depth_loss):
            raise RuntimeError(
                f"Non-finite depth loss at iteration {args.iteration}: "
                f"{depth_loss.item()}"
            )
        _backward(depth_loss, scaler, args.use_amp)
        del depth_predictions, depth_pred, depth_masks

        with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
            pose_predictions = forward_aggregator_heads(
                model,
                images,
                predict_depth=False,
                predict_camera=True,
            )
            abs_pose_loss = pose_loss_fn(pose_predictions["pose_enc"], poses)
            if use_rel_pose:
                rel_pose_loss = rel_pose_loss_fn(
                    pose_predictions["pose_enc"], poses
                )
            else:
                rel_pose_loss = abs_pose_loss.new_zeros(())
            pose_objective = (
                args.pose_weight * abs_pose_loss
                + args.rel_pose_weight * rel_pose_loss
            )
        if not torch.isfinite(pose_objective):
            raise RuntimeError(
                f"Non-finite pose loss at iteration {args.iteration}: "
                f"{pose_objective.item()}"
            )
        _backward(pose_objective, scaler, args.use_amp)
        total_loss_value = (
            float(depth_loss.detach())
            + args.pose_weight * float(abs_pose_loss.detach())
            + args.rel_pose_weight * float(rel_pose_loss.detach())
        )
    else:
        with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
            predictions = forward_aggregator_heads(model, images)
            depth_pred, depth_masks = align_depth_to_gt(
                predictions["depth"], depths, valid_masks
            )
            depth_loss = depth_loss_fn(depth_pred, depths, depth_masks)
            abs_pose_loss = pose_loss_fn(predictions["pose_enc"], poses)
            if use_rel_pose:
                rel_pose_loss = rel_pose_loss_fn(predictions["pose_enc"], poses)
            else:
                rel_pose_loss = abs_pose_loss.new_zeros(())
            total_loss = (
                depth_loss
                + args.pose_weight * abs_pose_loss
                + args.rel_pose_weight * rel_pose_loss
            )
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite fused loss at iteration {args.iteration}: "
                f"{total_loss.item()}"
            )
        _backward(total_loss, scaler, args.use_amp)
        total_loss_value = float(total_loss.detach())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_amp:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        args.gradient_clip_norm,
    )
    if args.use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
        raise RuntimeError(
            "Selected blocks received no gradient. Choose a block connected to an "
            "active depth/camera scale (recommended: --block_indices 17)."
        )

    scheduler.step()
    return {
        "total": total_loss_value,
        "depth": float(depth_loss.detach()),
        "abs_pose": float(abs_pose_loss.detach()),
        "rel_pose": float(rel_pose_loss.detach()),
        "grad_norm": float(grad_norm),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, iteration, loss_dict, args, path):
    checkpoint = {
        "format": "gct_aggregator_only_v2_fused",
        "iteration": iteration,
        "aggregator_state_dict": aggregator_state_dict(
            model,
            args.block_indices,
            args.train_frame_blocks,
            args.train_global_blocks,
        ),
        "loss_dict": loss_dict,
        "args": {
            key: value for key, value in vars(args).items()
            if key != "block_indices"
        },
        "block_indices": args.block_indices,
    }
    if args.save_optimizer_state:
        checkpoint.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        })
    torch.save(checkpoint, path)
    state_label = "weights + optimizer" if args.save_optimizer_state else "weights only"
    print(f"[checkpoint] Saved aggregator checkpoint ({state_label}) to {path}")


def plot_loss_history(history, output_dir):
    if not history["iteration"]:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    curves = [
        ("total", "Fused Total Loss"),
        ("depth", "Metric Depth Loss"),
        ("abs_pose", "Absolute Pose Loss"),
        ("rel_pose", "Relative Pose Loss"),
        ("grad_norm", "Gradient Norm"),
        ("lr", "Learning Rate"),
    ]
    for axis, (key, title) in zip(axes.flat, curves):
        axis.plot(history["iteration"], history[key], linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.3)
    axes[1, 2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


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


def parse_args():
    parser = argparse.ArgumentParser(description="GCT aggregator-only Stage1 training")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=2)
    parser.add_argument(
        "--sampler_type",
        choices=["temporal_nearby", "spatial_nearby"],
        default="spatial_nearby",
    )
    parser.add_argument("--spatial_radius", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_iterations", type=int, default=5000)
    parser.add_argument(
        "--block_indices",
        default="17",
        help="Block expression to random-init/train, e.g. 12-17 or 4,11,17",
    )
    parser.add_argument(
        "--train_last_n_blocks",
        type=int,
        default=None,
        help="Alternative to --block_indices; train the last N block pairs",
    )
    parser.add_argument("--frame_only", action="store_true")
    parser.add_argument("--global_only", action="store_true")
    parser.add_argument("--no_random_init", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--pose_weight", type=float, default=0.1)
    parser.add_argument("--rel_pose_weight", type=float, default=0.05)
    parser.add_argument("--rel_pose_start_iter", type=int, default=1000)
    parser.add_argument(
        "--joint_head_backward",
        action="store_true",
        help=(
            "Retain depth and camera graphs together. The default recomputes the "
            "aggregator and accumulates both gradients to reduce peak VRAM."
        ),
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--no_geometric_aug", action="store_true")
    parser.add_argument("--no_color_aug", action="store_true")
    parser.add_argument("--no_co_jitter", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help="Include AdamW/scheduler/scaler state for resume; greatly increases file size",
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--memory_probe",
        action="store_true",
        help="Run without writing checkpoints or plots",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    args = parser.parse_args()

    if args.frame_only and args.global_only:
        parser.error("--frame_only and --global_only are mutually exclusive")
    if args.min_views < 2:
        parser.error("Aggregator validation requires at least 2 views")
    if args.max_views < args.min_views:
        parser.error("--max_views must be >= --min_views")

    args.train_frame_blocks = not args.global_only
    args.train_global_blocks = not args.frame_only
    args.use_amp = not args.no_amp
    args.split_head_backward = not args.joint_head_backward
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
    print("GCT Aggregator-Only Stage1 Training")
    print("=" * 72)
    print(f"  Original checkpoint: {args.checkpoint}")
    print(f"  Views:              {args.min_views}-{args.max_views}")
    print(f"  Block expression:   {args.block_indices}")
    print(f"  Train frame/global: {args.train_frame_blocks}/{args.train_global_blocks}")
    print(f"  Random init:        {not args.no_random_init}")
    print(
        "  Fused loss:         "
        f"depth + {args.pose_weight:g}*abs_pose + "
        f"{args.rel_pose_weight:g}*rel_pose "
        f"(start={args.rel_pose_start_iter})"
    )
    print(f"  Split head backward:{args.split_head_backward}")
    print(f"  AMP / SDPA:         {args.use_amp} / {args.use_sdpa}")
    print(f"  Save optimizer:     {args.save_optimizer_state}")
    print(f"  Output:             {output_dir}")
    print("=" * 72)

    model = load_depth_gct_model(args.checkpoint, args.device, args.use_sdpa)
    args.block_indices = selected_block_indices(
        model,
        block_indices=args.block_indices,
        train_last_n_blocks=args.train_last_n_blocks,
    )
    args.block_tag = block_indices_to_tag(args.block_indices)
    print(f"  Resolved blocks:    {args.block_indices}")
    if not args.no_random_init and not args.resume:
        random_init_selected_blocks(
            model,
            args.block_indices,
            args.train_frame_blocks,
            args.train_global_blocks,
            args.seed,
        )
    configure_trainable_blocks(
        model,
        args.block_indices,
        args.train_frame_blocks,
        args.train_global_blocks,
    )

    dataloader = create_replica_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        min_views=args.min_views,
        max_views=args.max_views,
        max_dim=518,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        sampler_type=args.sampler_type,
        spatial_radius=args.spatial_radius,
        color_jitter_prob=0.0 if args.no_color_aug else 0.9,
        spatial_rescale_range=None if args.no_geometric_aug else (0.8, 1.2),
        aspect_ratio_range=None if args.no_geometric_aug else (0.33, 1.0),
        co_jitter=not args.no_co_jitter,
    )
    if args.max_samples is not None:
        subset_size = min(args.max_samples, len(dataloader.dataset))
        dataset = torch.utils.data.Subset(dataloader.dataset, range(subset_size))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"[data] Restricted training to {subset_size} samples")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
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
    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

    start_iteration = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" not in resume_ckpt:
            raise RuntimeError(
                "Resume checkpoint has no optimizer state. Train with "
                "--save_optimizer_state to create resumable checkpoints."
            )
        missing, unexpected = model.load_state_dict(
            resume_ckpt["aggregator_state_dict"], strict=False
        )
        expected_prefixes = []
        for idx in args.block_indices:
            if args.train_frame_blocks:
                expected_prefixes.append(f"aggregator.frame_blocks.{idx}.")
            if args.train_global_blocks:
                expected_prefixes.append(f"aggregator.global_blocks.{idx}.")
        relevant_missing = [
            key for key in missing if key.startswith(tuple(expected_prefixes))
        ]
        if relevant_missing or unexpected:
            raise RuntimeError(
                f"Resume state mismatch: missing={relevant_missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        start_iteration = int(resume_ckpt["iteration"])
        print(f"[resume] Continuing after iteration {start_iteration}")

    history = {
        "iteration": [],
        "total": [],
        "depth": [],
        "abs_pose": [],
        "rel_pose": [],
        "grad_norm": [],
        "lr": [],
        "max_allocated_gib": [],
        "max_reserved_gib": [],
    }
    data_iter = iter(dataloader)
    start_time = time.time()
    loss_dict = {}

    for iteration in range(start_iteration, args.total_iterations):
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
        history["total"].append(loss_dict["total"])
        history["depth"].append(loss_dict["depth"])
        history["abs_pose"].append(loss_dict["abs_pose"])
        history["rel_pose"].append(loss_dict["rel_pose"])
        history["grad_norm"].append(loss_dict["grad_norm"])
        history["lr"].append(current_lr)
        history["max_allocated_gib"].append(max_allocated)
        history["max_reserved_gib"].append(max_reserved)

        if completed == 1 or completed % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[Iter {completed}/{args.total_iterations}] "
                f"total={loss_dict['total']:.5f} "
                f"depth={loss_dict['depth']:.5f} "
                f"abs_pose={loss_dict['abs_pose']:.5f} "
                f"rel_pose={loss_dict['rel_pose']:.5f} "
                f"grad={loss_dict['grad_norm']:.4f} "
                f"lr={current_lr:.2e} "
                f"max_mem={max_allocated:.2f}/{max_reserved:.2f} GiB alloc/reserved "
                f"time={elapsed:.1f}s"
            )

        if (
            not args.memory_probe
            and args.save_every > 0
            and completed % args.save_every == 0
        ):
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

    if not args.memory_probe:
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
        with open(vis_dir / "loss_history.json", "w") as handle:
            json.dump(history, handle, indent=2)
        plot_loss_history(history, vis_dir)

    print("=" * 72)
    print("Training complete")
    print(f"  Final fused loss: {loss_dict.get('total', float('nan')):.6f}")
    print(f"  Final depth loss: {loss_dict.get('depth', float('nan')):.6f}")
    print(f"  Final abs pose:   {loss_dict.get('abs_pose', float('nan')):.6f}")
    print(f"  Final rel pose:   {loss_dict.get('rel_pose', float('nan')):.6f}")
    if history["max_allocated_gib"]:
        print(f"  Peak allocated:   {max(history['max_allocated_gib']):.2f} GiB")
        print(f"  Peak reserved:    {max(history['max_reserved_gib']):.2f} GiB")
    print("=" * 72)


if __name__ == "__main__":
    main()
