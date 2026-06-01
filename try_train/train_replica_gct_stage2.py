"""
GCTStream Head-Only Stage2 Training Script (论文4.2 对齐版)

将 GCT head-only 框架（aggregator 冻结 + 只训练 depth_head + camera_head）
迁移到 Stage2 长序列训练范式：

  - 从 Stage1 GCT checkpoint 初始化
  - Foldback Video Sampler（边界反向继续）
  - Progressive View Curriculum（views 数线性增长，如 8 -> 24）
  - Local Window Relative Pose Loss（窗口 k 在 [k_min, k_max] 随机采样）
  - AdamW lr=1e-4, wd=0.05, 5% warmup + cosine decay

模型架构沿用 train_replica_gct.py：
  GCTStream = DINOv2 ViT-L/14 + AggregatorStream + CameraCausalHead + DPTHead
  冻结 aggregator (~909M)，只训练 depth_head + camera_head (~281M)

用法：
  python try_train/train_replica_gct_stage2.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --stage1_checkpoint try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt \
    --output_dir try_train/checkpoints/gct_stage2_5k \
    --total_iterations 5000

  # Smoke test:
  python try_train/train_replica_gct_stage2.py ... --total_iterations 10 --log_every 2
"""

import os
import sys
import argparse
import time
import json
import random
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_stream import GCTStream
from foldback_video_sampler import (
    FoldbackVideoSampler,
    ProgressiveViewCurriculum,
    LocalWindowSampler,
)
from head_only_model import DepthLoss, PoseLoss
from train_replica_stage2 import (
    LocalRelativePoseLoss,
    ReplicaLongSequenceDataset,
)
from train_replica_gct import (
    load_gct_model,
    freeze_aggregator,
    forward_gct_heads_only,  # Stage1 global-attention forward (not used here; kept for reference)
    align_depth_to_gt,
)


DEPTH_MIN = 1e-4
DEPTH_MAX = 100.0


def _assert_trainable_params_finite(model, context):
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite trainable parameter detected after {context}: {name}")


def _first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        if torch.isfinite(grad).all():
            continue
        return name, int(torch.isnan(grad).sum().item()), int(torch.isinf(grad).sum().item())
    return None


# ---------------------------------------------------------------------------
# Stage2 GCA streaming forward (论文 4.2 对齐版)
# ---------------------------------------------------------------------------

def forward_gct_heads_only_streaming(model, images, sliding_window_size, num_frame_for_scale=8):
    """
    Stage2 / Streaming Model Forward (论文 4.2)

    与 Stage1 (`forward_gct_heads_only`) 的核心差异：
    | 维度                  | Stage1 (global attention)   | Stage2 (GCA streaming)         |
    |-----------------------|------------------------------|--------------------------------|
    | sliding_window_size   | -1 (无窗口)                  | k ∈ [k_min, k_max] (GCA 窗口) |
    | num_frame_per_block   | S (一次处理全部 views)        | 1 (逐帧 causal streaming)      |
    | num_frame_for_scale   | S (所有帧 bidirectional)     | 8 (仅前 8 帧 bidirectional)    |
    | causal_inference      | False                        | True (camera_head 用 KV cache) |

    论文 4.2 把第一阶段的 global attention 替换为 GCA（Geometric Context Attention），
    其本质是带局部窗口的因果注意力。Q/K/V projection 参数与 Stage1 一致，因此 Stage1
    checkpoint 可直接迁移到 Stage2 GCA 前向。

    Args:
        model: GCTStream model (aggregator frozen, heads trainable)
        images: [B, S, 3, H, W] in [0,1] range
        sliding_window_size: 当前 iter 的 GCA 局部窗口 k（来自 LocalWindowSampler）
        num_frame_for_scale: 前 N 帧 bidirectional 作 scale 估计 (默认 8，对应论文)

    Returns:
        dict with 'depth', 'depth_conf', 'pose_enc', 'pose_enc_list', 'images'
    """
    B, S = images.shape[:2]

    # scale 帧不能多于实际 view 数
    scale_frames = min(num_frame_for_scale, S)

    # Step 1: Aggregator forward — GCA streaming (no grad, saves memory)
    with torch.no_grad():
        model.clean_kv_cache()
        aggregated_tokens_list, patch_start_idx = model.aggregator(
            images,
            selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=scale_frames,     # Stage2: 前 N 帧 bidirectional
            sliding_window_size=sliding_window_size,  # Stage2: GCA 局部窗口
            num_frame_per_block=1,                # Stage2: 逐帧 streaming
        )
        model.clean_kv_cache()

    aggregated_tokens_list = [t.detach() for t in aggregated_tokens_list]

    # Step 2: Camera head forward — causal streaming inference with KV cache
    model.camera_head.clean_kv_cache()
    camera_output = model._predict_camera(
        aggregated_tokens_list,
        causal_inference=True,                    # Stage2: streaming + KV cache
        num_frame_per_block=1,
        num_frame_for_scale=scale_frames,
        sliding_window_size=sliding_window_size,  # 与 aggregator 一致的 GCA 窗口
    )
    model.camera_head.clean_kv_cache()

    # Step 3: Depth head forward (with grad) — DPT 本身不区分 stage
    depth_output = model._predict_depth(
        aggregated_tokens_list,
        images=images,
        patch_start_idx=patch_start_idx,
    )

    result = {}
    result.update(camera_output)
    result.update(depth_output)
    result['images'] = images
    return result


# ---------------------------------------------------------------------------
# Stage1 checkpoint loading
# ---------------------------------------------------------------------------

def load_gct_from_stage1(stage1_checkpoint, device, use_sdpa=True):
    """Initialize GCTStream from Stage1 GCT checkpoint."""
    ckpt = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=400, kv_cache_sliding_window=64,
        kv_cache_scale_frames=8, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=use_sdpa,
    )
    state_dict = ckpt.get('full_model_state_dict', ckpt.get('model_state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"[Stage1 init] Loaded from {stage1_checkpoint}")
    print(f"  Iteration: {ckpt.get('iteration', -1)}")
    print(f"  Loss dict: {ckpt.get('loss_dict', {})}")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    return model


# ---------------------------------------------------------------------------
# Anchor-scale normalization (论文 4.2)
# ---------------------------------------------------------------------------

def compute_anchor_scale(depths, valid_masks, poses, num_anchor_frames, source='depth_median'):
    """
    Compute anchor-scale s for normalizing depth + translation (论文 4.2 描述).

    论文用 anchor frames（前 N 帧）的点云尺度建立坐标系尺度，把 GT depth/translation
    除以 s 后再监督 head 输出，让不同 sample 的 loss 在 ~1 量级。

    Args:
        depths:        [B, V, H, W]  GT depth in meters
        valid_masks:   [B, V, H, W]  bool
        poses:         [B, V, 4, 4]  GT c2w pose (already reference-normalized)
        num_anchor_frames: N (use first N views as anchors)
        source: 'depth_median' (论文推荐) or 'translation_norm' (备选)

    Returns:
        s: [B] tensor; per-sample scale (clamped to >= 1e-3 to avoid divide-by-zero)
    """
    B, V = depths.shape[:2]
    N = min(num_anchor_frames, V)
    s = torch.ones(B, device=depths.device, dtype=depths.dtype)

    for b in range(B):
        if source == 'depth_median':
            # Use median depth of anchor frames as point-cloud scale proxy
            anc_depth = depths[b, :N]
            anc_mask = valid_masks[b, :N].bool()
            valid = anc_depth[anc_mask]
            if valid.numel() < 10:
                continue
            s_val = valid.float().median().clamp(min=1e-3)
            s[b] = s_val
        elif source == 'translation_norm':
            # Mean ||t_i|| over anchor frames (skip i=0 which is identity)
            if N <= 1:
                continue
            translations = poses[b, 1:N, :3, 3]  # [N-1, 3]
            norms = translations.float().norm(dim=-1)
            mean_norm = norms.mean().clamp(min=1e-3)
            s[b] = mean_norm
        else:
            raise ValueError(f"Unknown anchor scale source: {source}")

    return s


def apply_anchor_scale_normalization(depths, poses, s):
    """
    Normalize GT depth and pose translation by anchor scale s.
    Rotation is NOT scaled (scale-invariant).

    Args:
        depths: [B, V, H, W]
        poses:  [B, V, 4, 4]
        s:      [B] anchor scale

    Returns:
        depths_norm: [B, V, H, W]   = depths / s
        poses_norm:  [B, V, 4, 4]   t' = t / s, R unchanged
    """
    B = depths.shape[0]
    s_view = s.view(B, 1, 1, 1)
    depths_norm = depths / s_view

    poses_norm = poses.clone()
    s_view_pose = s.view(B, 1, 1)  # broadcast over (V, 3)
    poses_norm[:, :, :3, 3] = poses[:, :, :3, 3] / s_view_pose
    return depths_norm, poses_norm


# ---------------------------------------------------------------------------
# Training iteration (Stage2 with local window rel pose loss)
# ---------------------------------------------------------------------------

def train_one_iteration_gct_stage2(
    model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
    iteration, args, window_pairs, sliding_window_size, scaler=None
):
    """Train single iteration for GCT Stage2 (GCA streaming forward)."""
    # heads trainable, aggregator stays frozen
    model.depth_head.train()
    model.camera_head.train()
    model.aggregator.eval()

    images = batch['images'].to(args.device)
    depths = batch['depths'].to(args.device)
    valid_masks = batch['valid_masks'].to(args.device)
    poses = batch['poses'].to(args.device)

    # Anchor-scale normalization (论文 4.2): 把 GT depth/translation 除以
    # anchor frames 推得的 scale s，让不同 sample 的 loss 在 ~1 量级。
    # rotation 不缩放（scale-invariant）。
    anchor_s = None
    if args.use_anchor_scale_norm:
        anchor_s = compute_anchor_scale(
            depths, valid_masks, poses,
            num_anchor_frames=args.num_frame_for_scale,
            source=args.anchor_scale_source,
        )
        depths, poses = apply_anchor_scale_normalization(depths, poses, anchor_s)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        # Stage2 GCA streaming forward: 用 sampled sliding_window_size (= k)
        predictions = forward_gct_heads_only_streaming(
            model, images,
            sliding_window_size=sliding_window_size,
            num_frame_for_scale=args.num_frame_for_scale,
        )

        # Align depth resolution
        depth_pred, valid_masks_aligned = align_depth_to_gt(
            predictions['depth'], depths, valid_masks
        )

        if not torch.isfinite(depth_pred).all():
            optimizer.zero_grad(set_to_none=True)
            return {'total': float('nan'), 'depth': float('nan'), 'skipped': 1.0}

        depth_pred = depth_pred.float().clamp(min=DEPTH_MIN, max=DEPTH_MAX)
        depths_safe = torch.nan_to_num(
            depths.float(), nan=1.0, posinf=DEPTH_MAX, neginf=DEPTH_MIN
        ).clamp(min=DEPTH_MIN, max=DEPTH_MAX)

        # Depth loss (on normalized scale if anchor norm enabled)
        depth_loss = depth_loss_fn(depth_pred, depths_safe, valid_masks_aligned)
        loss = depth_loss
        loss_dict = {'depth': depth_loss.item()}
        if anchor_s is not None:
            loss_dict['anchor_s'] = float(anchor_s.mean().item())
        loss_dict['depth_min'] = float(depth_pred.detach().amin().item())
        loss_dict['depth_max'] = float(depth_pred.detach().amax().item())

        pose_enc = None
        if 'pose_enc' in predictions:
            if not torch.isfinite(predictions['pose_enc']).all():
                optimizer.zero_grad(set_to_none=True)
                return {'total': float('nan'), 'depth': loss_dict['depth'], 'skipped': 1.0}
            pose_enc = predictions['pose_enc'].float()

        # Absolute pose loss
        if pose_loss_fn is not None and pose_enc is not None and args.pose_weight != 0:
            pose_loss = pose_loss_fn(pose_enc, poses.float())
            loss = loss + args.pose_weight * pose_loss
            loss_dict['abs_pose'] = pose_loss.item()

        # Local window relative pose loss
        if rel_pose_loss_fn is not None and pose_enc is not None and args.rel_pose_weight != 0:
            if iteration >= args.rel_pose_start_iter and len(window_pairs) > 0:
                rel_pose_loss = rel_pose_loss_fn(pose_enc, poses.float(), window_pairs)
                rel_val = rel_pose_loss.item() if torch.is_tensor(rel_pose_loss) else float(rel_pose_loss)
                loss = loss + args.rel_pose_weight * rel_pose_loss
                loss_dict['rel_pose'] = rel_val
            else:
                loss_dict['rel_pose'] = 0.0

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        loss_dict['total'] = float('nan')
        loss_dict['skipped'] = 1.0
        return loss_dict

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_amp and scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            scaler.update()
        else:
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None:
                print(f"[nonfinite_grad] {bad_grad[0]} nan={bad_grad[1]} inf={bad_grad[2]}")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            loss_dict['skipped'] = 1.0
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None:
                print(f"[nonfinite_grad] {bad_grad[0]} nan={bad_grad[1]} inf={bad_grad[2]}")
            optimizer.zero_grad(set_to_none=True)
            loss_dict['skipped'] = 1.0

    _assert_trainable_params_finite(model, f"iteration {iteration}")

    if scheduler is not None:
        scheduler.step()

    loss_dict['total'] = loss.item()
    loss_dict['grad_norm'] = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
    loss_dict.setdefault('skipped', 0.0)
    return loss_dict


# ---------------------------------------------------------------------------
# Checkpoint + loss curves
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args, path):
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


def _plot_loss_curves(loss_history, vis_dir):
    iters = loss_history['iterations']
    if len(iters) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(iters, loss_history['total'], 'b-', linewidth=0.8)
    axes[0, 0].set_title('Total Loss'); axes[0, 0].set_xlabel('Iteration'); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, loss_history['depth'], 'g-', linewidth=0.8)
    axes[0, 1].set_title('Depth Loss'); axes[0, 1].set_xlabel('Iteration'); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(iters, loss_history['abs_pose'], 'r-', linewidth=0.8)
    axes[0, 2].set_title('Abs Pose Loss'); axes[0, 2].set_xlabel('Iteration'); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, loss_history['rel_pose'], 'm-', linewidth=0.8)
    axes[1, 0].set_title('Local Window Rel Pose Loss'); axes[1, 0].set_xlabel('Iteration'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, loss_history['lr'], 'k-', linewidth=0.8)
    axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

    if 'views' in loss_history and len(loss_history['views']) > 0:
        axes[1, 2].plot(iters, loss_history['views'], 'c-', linewidth=0.8)
        axes[1, 2].set_title('Views per Sample (curriculum)')
    axes[1, 2].set_xlabel('Iteration'); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = vis_dir / 'loss_curves.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[loss_curves] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCTStream Stage2 Head-Only Training (论文 4.2 对齐版)")

    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help="Stage1 GCT checkpoint path (full_model_state_dict)")

    # Model backend
    parser.add_argument('--use_sdpa', action='store_true', default=True)
    parser.add_argument('--use_flashinfer', dest='use_sdpa', action='store_false')

    # Image
    parser.add_argument('--img_size', type=int, default=518,
                        help="518 native resolution (lower if OOM on long sequences)")

    # Training params (paper 4.2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iterations', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Conservative GCT head fine-tuning LR; paper 4.2 used 5e-4 at larger scale")
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--pose_weight', type=float, default=0.1)
    parser.add_argument('--rel_pose_weight', type=float, default=0.05)
    parser.add_argument('--rel_pose_start_iter', type=int, default=500)

    # View curriculum (paper 24->320; pilot uses smaller range)
    parser.add_argument('--views_start', type=int, default=4,
                        help="Start views (paper=24, pilot for 518 + GCT mem)")
    parser.add_argument('--views_end', type=int, default=8,
                        help="End views (paper=320; pilot range due to 518 mem)")
    parser.add_argument('--warmup_iterations', type=int, default=1000,
                        help="Iterations during which view count doesn't grow")

    # Local window (paper [16,64]; pilot smaller)
    parser.add_argument('--k_min', type=int, default=2,
                        help="Min local window k (paper=16; pilot smaller given fewer views)")
    parser.add_argument('--k_max', type=int, default=4,
                        help="Max local window k (paper=64)")
    parser.add_argument('--num_frame_for_scale', type=int, default=8,
                        help="Stage2 bidirectional scale frames (paper=8); clamped to S at runtime")

    # Anchor-scale normalization (论文 4.2 / implementation_checklist_status)
    parser.add_argument('--use_anchor_scale_norm', action='store_true', default=True,
                        help="Normalize GT depth + translation by anchor scale before loss (default: True)")
    parser.add_argument('--no_anchor_scale_norm', dest='use_anchor_scale_norm', action='store_false',
                        help="Disable anchor-scale normalization (fall back to pure metric loss)")
    parser.add_argument('--anchor_scale_source', type=str, default='depth_median',
                        choices=['depth_median', 'translation_norm'],
                        help="Source for anchor scale: depth_median (论文推荐) or translation_norm")

    # Foldback sampler
    parser.add_argument('--stride_min', type=int, default=1)
    parser.add_argument('--stride_max', type=int, default=3)

    # LR scheduler
    parser.add_argument('--lr_warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-8)

    # Other
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gct_stage2')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--skip_final_save', action='store_true')
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_warmup_iterations = max(1, int(args.total_iterations * args.lr_warmup_ratio))

    print(f"\n{'='*60}")
    print(f"GCTStream Stage2 Head-Only Training (论文 4.2 对齐版)")
    print(f"{'='*60}")
    print(f"  - Model: GCTStream (full), Aggregator frozen, Heads trainable")
    print(f"  - Resolution: {args.img_size}")
    print(f"  - lr: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"  - total_iterations: {args.total_iterations}")
    print(f"  - views curriculum: {args.views_start} -> {args.views_end}")
    print(f"  - local window k: [{args.k_min}, {args.k_max}]")
    print(f"  - anchor-scale norm: {args.use_anchor_scale_norm} (source={args.anchor_scale_source}, N={args.num_frame_for_scale})")
    print(f"  - stage1 checkpoint: {args.stage1_checkpoint}")
    print(f"Data: {args.data_root}")

    # 1. Detect total frames
    traj_file = Path(args.data_root) / "traj.txt"
    with open(traj_file, 'r') as f:
        total_frames = len([l for l in f.readlines() if len(l.strip().split()) == 16])
    print(f"  - Total frames detected: {total_frames}")

    # 2. Foldback sampler
    foldback_sampler = FoldbackVideoSampler(
        total_frames=total_frames,
        stride_range=(args.stride_min, args.stride_max),
        redraw_stride_after_reverse=True,
        seed=args.seed,
    )

    # 3. View curriculum
    view_curriculum = ProgressiveViewCurriculum(
        views_start=args.views_start,
        views_end=args.views_end,
        total_iterations=args.total_iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
    )

    # 4. Local window sampler
    window_sampler = LocalWindowSampler(
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )

    # 5. Dataset
    dataset = ReplicaLongSequenceDataset(
        data_root=args.data_root,
        foldback_sampler=foldback_sampler,
        max_dim=args.img_size,
        seed=args.seed,
    )

    # 6. DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 7. Load model from Stage1 GCT checkpoint
    model = load_gct_from_stage1(args.stage1_checkpoint, args.device, use_sdpa=args.use_sdpa)
    freeze_aggregator(model)

    # 8. Loss functions
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = LocalRelativePoseLoss()

    # 9. Optimizer (only heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[Optimizer] Trainable parameters: {trainable_count:,}")

    # 10. Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.min_lr / args.lr,
        end_factor=1.0,
        total_iters=lr_warmup_iterations
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.total_iterations - lr_warmup_iterations),
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[lr_warmup_iterations]
    )

    # 11. Scaler
    scaler = GradScaler() if args.use_amp else None

    # 12. Resume
    start_iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['full_model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        freeze_aggregator(model)
        print(f"[Resume] From iteration {start_iteration}")

    # 13. Training loop
    print(f"\n{'='*60}")
    print(f"Starting Stage2 GCT training")
    print(f"{'='*60}")

    loss_history = {
        'iterations': [], 'total': [], 'depth': [], 'abs_pose': [], 'rel_pose': [],
        'lr': [], 'views': [], 'grad_norm': [], 'skipped': [], 'depth_min': [], 'depth_max': [],
    }

    start_time = time.time()
    iteration = start_iteration
    data_iter = iter(train_dataloader)
    loss_dict = {'total': 0.0}

    while iteration < args.total_iterations:
        current_views = view_curriculum.get_num_views_with_variance(iteration, variance=2)
        current_views = max(2, current_views)
        dataset.set_num_views(current_views)

        k = window_sampler.sample_window_size()
        window_pairs = window_sampler.get_adjacent_pairs(current_views, k)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        loss_dict = train_one_iteration_gct_stage2(
            model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
            iteration, args, window_pairs, sliding_window_size=k, scaler=scaler
        )

        iteration += 1

        if iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            anchor_s_str = f", anchor_s: {loss_dict.get('anchor_s', 0):.3f}" if 'anchor_s' in loss_dict else ""
            grad_norm_str = f", grad_norm: {loss_dict.get('grad_norm', 0):.3f}" if 'grad_norm' in loss_dict else ""
            skipped_str = f", skipped: {int(loss_dict.get('skipped', 0))}"
            print(f"[Iter {iteration}/{args.total_iterations}] "
                  f"Views: {current_views}, k: {k} "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(depth: {loss_dict.get('depth', 0):.4f}, "
                  f"abs_pose: {loss_dict.get('abs_pose', 0):.4f}, "
                  f"rel_pose: {loss_dict.get('rel_pose', 0):.4f}{anchor_s_str}"
                  f"{grad_norm_str}{skipped_str}) "
                  f"LR: {current_lr:.2e} "
                  f"Time: {elapsed:.1f}s")

            loss_history['iterations'].append(iteration)
            loss_history['total'].append(loss_dict['total'])
            loss_history['depth'].append(loss_dict.get('depth', 0))
            loss_history['abs_pose'].append(loss_dict.get('abs_pose', 0))
            loss_history['rel_pose'].append(loss_dict.get('rel_pose', 0))
            loss_history['lr'].append(current_lr)
            loss_history['views'].append(current_views)
            loss_history['grad_norm'].append(loss_dict.get('grad_norm', 0))
            loss_history['skipped'].append(loss_dict.get('skipped', 0))
            loss_history['depth_min'].append(loss_dict.get('depth_min', 0))
            loss_history['depth_max'].append(loss_dict.get('depth_max', 0))

        if iteration % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                            output_dir / f'checkpoint_iter_{iteration}.pt')

        if iteration % 100 == 0:
            torch.cuda.empty_cache()

    if args.skip_final_save:
        print("[save_checkpoint] Skipped final checkpoint (--skip_final_save)")
    else:
        save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                        output_dir / 'checkpoint_final.pt')

    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    with open(vis_dir / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"[loss_history] Saved to {vis_dir / 'loss_history.json'}")
    _plot_loss_curves(loss_history, vis_dir)

    print(f"\n{'='*60}")
    print(f"GCTStream Stage2 Training completed")
    print(f"{'='*60}")
    print(f"  - Total iterations: {iteration}")
    print(f"  - Final loss: {loss_dict['total']:.4f}")
    print(f"  - Checkpoint saved to: {output_dir}")


if __name__ == '__main__':
    main()
