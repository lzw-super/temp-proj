"""Train LZW-Map Stage1 with a frozen DINOv2 backbone.

The student is created from scratch:
  - aggregator.patch_embed loads pretrained ViT-B/14 reg weights from the
    original local DINOv2 repository and is frozen.
  - aggregator frame/global blocks, camera head, and depth head stay randomly
    initialized and are trained with the Stage1 fused objective.

Usage:
  python try_train/train_lzw_map_stage1.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --dinov2_repo /home/lizhengwu/desktop/temp_proj/dinov2 \
    --output_dir try_train/checkpoints/lzw_map_stage1_smoke \
    --total_iterations 50
"""

import argparse
import gc
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

from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from replica_dataset import create_replica_dataloader
from lingbot_map.models.lzw_map import (
    create_lzw_map_stage1,
    format_parameter_rows,
    lzw_map_parameter_summary,
)


DEFAULT_DINOV2_REPO = "/home/lizhengwu/desktop/temp_proj/dinov2"


def _compatible_state_dict(source_state, target_state):
    compatible = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append((key, "missing_in_target"))
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped.append((key, f"{tuple(value.shape)} != {tuple(target_state[key].shape)}"))
            continue
        compatible[key] = value
    return compatible, skipped


def load_backbone_from_original_dinov2_repo(model, dinov2_repo, hub_name):
    """Load only aggregator.patch_embed from the original DINOv2 repo."""
    if not Path(dinov2_repo).exists():
        raise FileNotFoundError(f"DINOv2 repo not found: {dinov2_repo}")

    print(f"[dinov2] Loading {hub_name} from original repo: {dinov2_repo}")
    dino_model = torch.hub.load(
        dinov2_repo,
        hub_name,
        source="local",
        pretrained=True,
    )
    source_state = dino_model.state_dict()
    target_state = model.aggregator.patch_embed.state_dict()
    compatible, skipped = _compatible_state_dict(source_state, target_state)

    missing, unexpected = model.aggregator.patch_embed.load_state_dict(
        compatible,
        strict=False,
    )
    del dino_model, source_state, target_state, compatible
    gc.collect()

    loaded_params = sum(
        tensor.numel()
        for name, tensor in model.aggregator.patch_embed.state_dict().items()
        if name not in set(missing)
    )
    print(f"[dinov2] Loaded compatible tensors into frozen backbone")
    print(f"[dinov2] Missing keys after load: {len(missing)}")
    print(f"[dinov2] Unexpected keys after load: {len(unexpected)}")
    print(f"[dinov2] Skipped source tensors: {len(skipped)}")
    if skipped[:5]:
        print(f"[dinov2] First skipped tensors: {skipped[:5]}")
    print(f"[dinov2] Backbone state elements after load: {loaded_params:,}")


def freeze_backbone_train_rest(model):
    """Freeze only the DINOv2 patch_embed backbone; train all Stage1 modules after it."""
    for param in model.parameters():
        param.requires_grad = True

    for param in model.aggregator.patch_embed.parameters():
        param.requires_grad = False
    if hasattr(model.aggregator.patch_embed, "mask_token"):
        model.aggregator.patch_embed.mask_token.requires_grad_(False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"[freeze] Trainable params: {trainable:,} ({trainable / total:.2%})")
    print(f"[freeze] Frozen backbone params: {frozen:,} ({frozen / total:.2%})")


def set_training_modes(model):
    model.train()
    model.aggregator.patch_embed.eval()


def forward_lzw_stage1(model, images):
    """Stage1 global-attention forward for LZW-Map."""
    _, num_views = images.shape[:2]

    model.clean_kv_cache()
    features, patch_start_idx = model.aggregator(
        images,
        selected_idx=model.selected_idx,
        num_frame_for_scale=num_views,
        sliding_window_size=-1,
        num_frame_per_block=num_views,
    )
    model.clean_kv_cache()

    model.camera_head.clean_kv_cache()
    camera_output = model._predict_camera(
        features,
        causal_inference=False,
        num_frame_per_block=num_views,
        num_frame_for_scale=num_views,
    )
    model.camera_head.clean_kv_cache()

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
    set_training_modes(model)

    images = batch["images"].to(args.device, non_blocking=True)
    depths = batch["depths"].to(args.device, non_blocking=True)
    valid_masks = batch["valid_masks"].to(args.device, non_blocking=True)
    poses = batch["poses"].to(args.device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_rel_pose = args.iteration >= args.rel_pose_start_iter

    with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
        predictions = forward_lzw_stage1(model, images)
        depth_pred, valid_masks_aligned = align_depth_to_gt(
            predictions["depth"],
            depths,
            valid_masks,
        )
        depth_loss = depth_loss_fn(depth_pred, depths, valid_masks_aligned)
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
            f"Non-finite loss at iteration {args.iteration}: {float(total_loss.detach())}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        args.gradient_clip_norm,
    )
    grad_is_finite = bool(torch.isfinite(grad_norm))

    skipped_step = False
    if args.use_amp:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_step = scaler.get_scale() < previous_scale
        if skipped_step:
            print(
                f"[amp] Iter {args.iteration}: skipped optimizer step, "
                f"grad_norm={float(grad_norm):.4f}, "
                f"loss scale {previous_scale:.1f}->{scaler.get_scale():.1f}"
            )
    else:
        if not grad_is_finite:
            raise RuntimeError(f"Non-finite gradient norm: {float(grad_norm)}")
        optimizer.step()

    if not skipped_step:
        scheduler.step()

    return {
        "total": float(total_loss.detach()),
        "depth": float(depth_loss.detach()),
        "abs_pose": float(abs_pose_loss.detach()),
        "rel_pose": float(rel_pose_loss.detach()),
        "grad_norm": float(grad_norm) if grad_is_finite else float("nan"),
    }


def trainable_state_dict(model):
    """Save the random-initialized/trainable LZW modules, excluding frozen DINO backbone."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("aggregator.patch_embed.")
    }


def save_checkpoint(model, optimizer, scheduler, scaler, iteration, loss_dict, args, path):
    checkpoint = {
        "format": "lzw_map_stage1_frozen_dinov2_backbone",
        "model_name": "lzw-map-stage1",
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
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        })
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def plot_loss_history(history, output_dir):
    if not history["iteration"]:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    curves = [
        ("total", "Total Loss"),
        ("depth", "Depth Loss"),
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
    parser = argparse.ArgumentParser(description="LZW-Map frozen-backbone Stage1 training")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dinov2_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--dinov2_hub_name", default="dinov2_vitb14_reg")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument(
        "--sampler_type",
        choices=["temporal_nearby", "spatial_nearby"],
        default="spatial_nearby",
    )
    parser.add_argument("--spatial_radius", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--pose_weight", type=float, default=0.1)
    parser.add_argument("--rel_pose_weight", type=float, default=0.05)
    parser.add_argument("--rel_pose_start_iter", type=int, default=500)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--no_geometric_aug", action="store_true")
    parser.add_argument("--no_color_aug", action="store_true")
    parser.add_argument("--no_co_jitter", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--save_full_model_state", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    args = parser.parse_args()

    if args.min_views < 2:
        parser.error("--min_views must be >= 2")
    if args.max_views < args.min_views:
        parser.error("--max_views must be >= --min_views")
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
    print("LZW-Map Stage1 Training (frozen original DINOv2 backbone)")
    print("=" * 72)
    print(f"  DINOv2 repo:        {args.dinov2_repo}")
    print(f"  DINOv2 hub name:    {args.dinov2_hub_name}")
    print(f"  Trainable modules:  aggregator frame/global blocks + camera/depth heads")
    print(f"  Frozen module:      aggregator.patch_embed")
    print(f"  Views:              {args.min_views}-{args.max_views}")
    print(f"  Sampler:            {args.sampler_type}")
    print(f"  LR / WD:            {args.lr:g} / {args.weight_decay:g}")
    print(
        "  Loss:               "
        f"depth + {args.pose_weight:g}*abs_pose + "
        f"{args.rel_pose_weight:g}*rel_pose "
        f"(start={args.rel_pose_start_iter})"
    )
    print(f"  AMP / SDPA:         {args.use_amp} / {args.use_sdpa}")
    print(f"  Output:             {output_dir}")
    print("=" * 72)

    model = create_lzw_map_stage1(
        pretrained_path="",
        enable_point=False,
        use_sdpa=args.use_sdpa,
    )
    load_backbone_from_original_dinov2_repo(
        model,
        args.dinov2_repo,
        args.dinov2_hub_name,
    )
    freeze_backbone_train_rest(model)
    model = model.to(args.device)

    print("[model] Parameter summary")
    for line in format_parameter_rows(lzw_map_parameter_summary(model)):
        print(line)

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = resume_ckpt.get("trainable_state_dict", resume_ckpt.get("full_model_state_dict"))
        if state is None:
            raise RuntimeError("Resume checkpoint has no trainable/full model state dict")
        missing, unexpected = model.load_state_dict(state, strict=False)
        relevant_missing = [
            key for key in missing
            if not key.startswith("aggregator.patch_embed.")
        ]
        if relevant_missing or unexpected:
            raise RuntimeError(
                f"Resume state mismatch: missing={relevant_missing[:8]}, unexpected={unexpected[:8]}"
            )
        print(f"[resume] Loaded model state from {args.resume}")

    if args.dry_run:
        print("[dry_run] Model constructed, DINOv2 backbone loaded/frozen. Exiting.")
        return

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

    start_iteration = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.save_optimizer_state and "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            if resume_ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        start_iteration = int(resume_ckpt.get("iteration", 0))
        print(f"[resume] Continuing after iteration {start_iteration}")

    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

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
    print(f"  Final total:      {loss_dict.get('total', float('nan')):.6f}")
    print(f"  Final depth:      {loss_dict.get('depth', float('nan')):.6f}")
    print(f"  Final abs pose:   {loss_dict.get('abs_pose', float('nan')):.6f}")
    print(f"  Final rel pose:   {loss_dict.get('rel_pose', float('nan')):.6f}")
    if history["max_allocated_gib"]:
        print(f"  Peak allocated:   {max(history['max_allocated_gib']):.2f} GiB")
        print(f"  Peak reserved:    {max(history['max_reserved_gib']):.2f} GiB")
    print("=" * 72)


if __name__ == "__main__":
    main()
