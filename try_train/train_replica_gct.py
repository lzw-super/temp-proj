"""
GCTStream Head-Only Stage1 Training Script

Based on the full GCTStream model (not HeadOnlyModel).
Freezes aggregator (DINOv2 ViT-L/14 + AggregatorStream), trains only depth_head + camera_head.

Architecture:
  GCTStream = DINOv2 ViT-L/14 + AggregatorStream + CameraCausalHead + DPTHead

Training strategy:
  1. Load pretrained GCTStream checkpoint
  2. Freeze aggregator (DINOv2 + frame_blocks + global_blocks)
  3. Run aggregator with torch.no_grad() to save memory
  4. Train only depth_head + camera_head with gradient tracking
  5. Loss: depth_loss + pose_weight * abs_pose_loss + rel_pose_weight * rel_pose_loss

Key differences from train_replica_v2.py (HeadOnlyModel):
  - Full GCTStream model with cross-frame attention (AggregatorStream)
  - Image resolution: 518 (ViT-L/14 native resolution)
  - CameraCausalHead with iterative refinement (4 iterations)
  - Memory-efficient: aggregator forward with no_grad, only heads track gradients

Usage:
  python try_train/train_replica_gct.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --checkpoint /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --output_dir try_train/checkpoints/gct_stage1

  # Smoke test (50 iterations):
  python try_train/train_replica_gct.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --checkpoint /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --output_dir try_train/checkpoints/gct_stage1_smoke \
    --total_iterations 50 --log_every 5
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_stream import GCTStream
from replica_dataset import create_replica_dataloader
from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss


# ---------------------------------------------------------------------------
# Model loading & freezing
# ---------------------------------------------------------------------------

def load_gct_model(checkpoint_path, device, use_sdpa=True):
    """Load GCTStream model from pretrained checkpoint."""
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=100,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    if missing:
        print(f"[load_gct_model] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[load_gct_model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    print(f"[load_gct_model] Loaded from {checkpoint_path}")
    return model


def freeze_aggregator(model):
    """Freeze aggregator parameters, keep heads trainable."""
    for param in model.aggregator.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"[freeze_aggregator] Trainable: {trainable:,} ({trainable/total:.2%})")
    print(f"[freeze_aggregator] Frozen: {frozen:,} ({frozen/total:.2%})")
    print(f"[freeze_aggregator] Total: {total:,}")


# ---------------------------------------------------------------------------
# Forward pass: aggregator no_grad + heads with grad
# ---------------------------------------------------------------------------

def forward_gct_heads_only(model, images):
    """
    Stage1 / Base Model Forward (论文 4.1)

    设计对应论文 Sec. 4.1 Stage1：**global attention**（无 GCA 窗口）+ 一次性处理所有 views
    （bidirectional + global）。等价于把整个 sample 视为一个 multi-view set。

    关键参数对应论文 4.1：
      - sliding_window_size = -1         (全局注意力，无 GCA 窗口)
      - num_frame_per_block = S          (一次处理全部 views，不是逐帧 streaming)
      - num_frame_for_scale = S          (所有帧都参与 scale 估计，bidirectional)
      - causal_inference = False         (camera_head 非 causal/streaming)

    aggregator 用 no_grad 前向以节省显存，只有 heads 追踪梯度。

    Args:
        model: GCTStream model (aggregator frozen, heads trainable)
        images: [B, S, 3, H, W] in [0,1] range

    Returns:
        dict with 'depth', 'depth_conf', 'pose_enc', 'pose_enc_list', 'images'
    """
    B, S = images.shape[:2]

    # Step 1: Aggregator forward (no grad, global attention)
    with torch.no_grad():
        model.clean_kv_cache()
        aggregated_tokens_list, patch_start_idx = model.aggregator(
            images,
            selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=S,         # Stage1: 全部帧都做 bidirectional
            sliding_window_size=-1,        # Stage1: 无 GCA 窗口（全局注意力）
            num_frame_per_block=S,         # Stage1: 一次处理全部 views
        )
        model.clean_kv_cache()

    # Detach features (already no_grad, but explicit for clarity)
    aggregated_tokens_list = [t.detach() for t in aggregated_tokens_list]

    # Step 2: Camera head forward (with grad) — Stage1 也是 global，非 causal
    model.camera_head.clean_kv_cache()
    camera_output = model._predict_camera(
        aggregated_tokens_list,
        causal_inference=False,            # Stage1: 非 streaming，所有帧 bidirectional
        num_frame_per_block=S,
        num_frame_for_scale=S,
    )
    model.camera_head.clean_kv_cache()

    # Step 3: Depth head forward (with grad)
    depth_output = model._predict_depth(
        aggregated_tokens_list,
        images=images,
        patch_start_idx=patch_start_idx,
    )

    # Combine results
    result = {}
    result.update(camera_output)
    result.update(depth_output)
    result['images'] = images

    return result


# ---------------------------------------------------------------------------
# Depth alignment (handle resolution mismatch between model output and GT)
# ---------------------------------------------------------------------------

def align_depth_to_gt(depth_pred, depth_gt, valid_mask_gt):
    """
    Align predicted depth to GT resolution if needed.

    Args:
        depth_pred: [B, S, H_pred, W_pred, 1]
        depth_gt: [B, S, H_gt, W_gt]
        valid_mask_gt: [B, S, H_gt, W_gt]

    Returns:
        depth_pred_aligned: [B, S, H_gt, W_gt]
        valid_mask: [B, S, H_gt, W_gt]
    """
    if depth_pred.dim() == 5:
        depth_pred = depth_pred.squeeze(-1)

    B, S, H_pred, W_pred = depth_pred.shape
    _, _, H_gt, W_gt = depth_gt.shape

    if H_pred != H_gt or W_pred != W_gt:
        depth_pred_flat = depth_pred.reshape(B * S, 1, H_pred, W_pred)
        depth_pred_resized = torch.nn.functional.interpolate(
            depth_pred_flat,
            size=(H_gt, W_gt),
            mode='bilinear',
            align_corners=False,
        )
        depth_pred = depth_pred_resized.reshape(B, S, H_gt, W_gt)

    return depth_pred, valid_mask_gt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_iteration(
    model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
    iteration, args, scaler=None
):
    """Train single iteration."""
    # Heads in train mode, aggregator stays in eval mode (frozen)
    model.depth_head.train()
    model.camera_head.train()
    model.aggregator.eval()

    images = batch['images'].to(args.device)
    depths = batch['depths'].to(args.device)
    valid_masks = batch['valid_masks'].to(args.device)
    poses = batch['poses'].to(args.device)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        predictions = forward_gct_heads_only(model, images)

        # Align depth resolution
        depth_pred, valid_masks_aligned = align_depth_to_gt(
            predictions['depth'], depths, valid_masks
        )

        # Depth loss
        depth_loss = depth_loss_fn(depth_pred, depths, valid_masks_aligned)
        loss = depth_loss
        loss_dict = {'depth': depth_loss.item()}

        # Absolute pose loss
        if pose_loss_fn is not None and 'pose_enc' in predictions:
            pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
            loss = loss + args.pose_weight * pose_loss
            loss_dict['abs_pose'] = pose_loss.item()

        # Relative pose loss (delayed start)
        if rel_pose_loss_fn is not None and 'pose_enc' in predictions:
            if iteration >= args.rel_pose_start_iter:
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses)
                loss = loss + args.rel_pose_weight * rel_pose_loss
                loss_dict['rel_pose'] = rel_pose_loss.item()
            else:
                loss_dict['rel_pose'] = 0.0

    if args.use_amp and scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    loss_dict['total'] = loss.item()

    return loss_dict


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------

def _plot_loss_curves(loss_history, vis_dir):
    """Plot and save loss curves."""
    iters = loss_history['iterations']
    if len(iters) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(iters, loss_history['total'], 'b-', linewidth=0.8)
    axes[0, 0].set_title('Total Loss'); axes[0, 0].set_xlabel('Iteration'); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, loss_history['depth'], 'g-', linewidth=0.8)
    axes[0, 1].set_title('Depth Loss (masked_log_l1)'); axes[0, 1].set_xlabel('Iteration'); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(iters, loss_history['abs_pose'], 'r-', linewidth=0.8)
    axes[0, 2].set_title('Absolute Pose Loss'); axes[0, 2].set_xlabel('Iteration'); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, loss_history['rel_pose'], 'm-', linewidth=0.8)
    axes[1, 0].set_title('Relative Pose Loss'); axes[1, 0].set_xlabel('Iteration'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, loss_history['lr'], 'k-', linewidth=0.8)
    axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

    for key, color, label in [('depth', 'g', 'depth'), ('abs_pose', 'r', 'abs_pose'), ('rel_pose', 'm', 'rel_pose')]:
        vals = loss_history[key]
        if max(vals) > 0:
            axes[1, 2].plot(iters, vals, color=color, linewidth=0.8, label=label, alpha=0.8)
    axes[1, 2].set_title('All Losses (overlay)'); axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = vis_dir / 'loss_curves.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[loss_curves] Saved to {save_path}")


def save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args, path):
    """Save training checkpoint (only head parameters for efficiency)."""
    head_state_dict = {
        k: v for k, v in model.state_dict().items()
        if k.startswith(('depth_head', 'camera_head'))
    }
    checkpoint = {
        'iteration': iteration,
        'head_state_dict': head_state_dict,
        'full_model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_dict': loss_dict,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"[save_checkpoint] Saved to {path} (iteration {iteration})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCTStream Head-Only Stage1 Training")

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help="Replica data root directory")

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Pretrained GCTStream checkpoint path")
    parser.add_argument('--use_sdpa', action='store_true', default=True,
                        help="Use SDPA backend for KV cache (default: True)")
    parser.add_argument('--use_flashinfer', dest='use_sdpa', action='store_false',
                        help="Use FlashInfer backend instead of SDPA")

    # View sampling
    parser.add_argument('--min_views', type=int, default=2)
    parser.add_argument('--max_views', type=int, default=2,
                        help="Max views per batch (keep <=4 for 518 resolution to fit GPU)")
    parser.add_argument('--sampler_type', type=str, default='temporal_nearby',
                        choices=['temporal_nearby', 'spatial_nearby'])
    parser.add_argument('--spatial_radius', type=float, default=5.0)

    # Training params (paper 4.1 settings)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iterations', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--pose_weight', type=float, default=0.1)
    parser.add_argument('--rel_pose_weight', type=float, default=0.05)
    parser.add_argument('--rel_pose_start_iter', type=int, default=1000)

    # Scheduler
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-8)

    # Geometric augmentation
    parser.add_argument('--no_geometric_aug', action='store_true')
    parser.add_argument('--no_co_jitter', action='store_true')

    # Other
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gct_stage1')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_iterations = int(args.total_iterations * args.warmup_ratio)

    print(f"\n{'='*60}")
    print(f"GCTStream Head-Only Stage1 Training")
    print(f"{'='*60}")
    print(f"  - Model: GCTStream (full), Aggregator frozen, Heads trainable")
    print(f"  - Checkpoint: {args.checkpoint}")
    print(f"  - Resolution: 518 (ViT-L/14 native)")
    print(f"  - lr: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"  - total_iterations: {args.total_iterations}")
    print(f"  - warmup: {warmup_iterations} ({args.warmup_ratio*100:.0f}%)")
    print(f"  - views: [{args.min_views}, {args.max_views}]")
    print(f"  - geometric_aug: {not args.no_geometric_aug}")
    print(f"  - use_sdpa: {args.use_sdpa}")
    print(f"Data: {args.data_root}")

    # 1. DataLoader (518 resolution for GCT model)
    train_dataloader = create_replica_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        min_views=args.min_views,
        max_views=args.max_views,
        max_dim=518,
        shuffle=True,
        num_workers=4,
        seed=args.seed,
        sampler_type=args.sampler_type,
        spatial_radius=args.spatial_radius,
        spatial_rescale_range=None if args.no_geometric_aug else (0.8, 1.2),
        aspect_ratio_range=None if args.no_geometric_aug else (0.33, 1.0),
        co_jitter=not args.no_co_jitter,
    )

    if args.max_samples is not None:
        limited_dataset = torch.utils.data.Subset(
            train_dataloader.dataset,
            range(min(args.max_samples, len(train_dataloader.dataset)))
        )
        train_dataloader = torch.utils.data.DataLoader(
            limited_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        print(f"[limit] Using {len(limited_dataset)} samples for quick test")

    # 2. Model
    model = load_gct_model(args.checkpoint, args.device, use_sdpa=args.use_sdpa)
    freeze_aggregator(model)

    # 3. Loss functions (reuse from head_only_model)
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

    # 4. Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[Optimizer] Trainable parameters: {trainable_count:,}")

    # 5. Scheduler (5% warmup + cosine decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.min_lr / args.lr,
        end_factor=1.0,
        total_iters=warmup_iterations
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.total_iterations - warmup_iterations,
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iterations]
    )

    # 6. Scaler
    scaler = GradScaler() if args.use_amp else None

    # 7. Resume
    start_iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['full_model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        # Re-freeze aggregator after loading
        freeze_aggregator(model)
        print(f"[Resume] From iteration {start_iteration}")

    # 8. Training loop
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")

    loss_history = {
        'iterations': [],
        'total': [],
        'depth': [],
        'abs_pose': [],
        'rel_pose': [],
        'lr': [],
    }

    start_time = time.time()
    iteration = start_iteration
    data_iter = iter(train_dataloader)

    while iteration < args.total_iterations:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        loss_dict = train_one_iteration(
            model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
            iteration, args, scaler
        )

        iteration += 1

        if iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Iter {iteration}/{args.total_iterations}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(depth: {loss_dict.get('depth', 0):.4f}, "
                  f"abs_pose: {loss_dict.get('abs_pose', 0):.4f}, "
                  f"rel_pose: {loss_dict.get('rel_pose', 0):.4f}) "
                  f"LR: {current_lr:.2e} "
                  f"Time: {elapsed:.1f}s")

            loss_history['iterations'].append(iteration)
            loss_history['total'].append(loss_dict['total'])
            loss_history['depth'].append(loss_dict.get('depth', 0))
            loss_history['abs_pose'].append(loss_dict.get('abs_pose', 0))
            loss_history['rel_pose'].append(loss_dict.get('rel_pose', 0))
            loss_history['lr'].append(current_lr)

        if iteration % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                           output_dir / f'checkpoint_iter_{iteration}.pt')

        # Periodic memory cleanup
        if iteration % 100 == 0:
            torch.cuda.empty_cache()

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                   output_dir / 'checkpoint_final.pt')

    # Save loss history and curves
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    loss_json_path = vis_dir / 'loss_history.json'
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"[loss_history] Saved to {loss_json_path}")

    _plot_loss_curves(loss_history, vis_dir)

    print(f"\n{'='*60}")
    print(f"Training completed")
    print(f"{'='*60}")
    print(f"  - Total iterations: {iteration}")
    print(f"  - Final loss: {loss_dict['total']:.4f}")
    print(f"  - Checkpoint saved to: {output_dir}")


if __name__ == '__main__':
    main()
