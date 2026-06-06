"""
GCTStream Backbone-Only Training Script

将 lingbot-map 原始 ViT-L backbone 替换为 DINOv2 ViT-B/S + adapter，
冻结原始 downstream（aggregator blocks + heads），只训练新 backbone + adapter。

Architecture:
  Hybrid GCTStream = DINOv2 ViT-B/14 + Linear(768->1024)
                   + original AggregatorStream/CameraCausalHead/DPTHead

Training strategy:
  1. 创建 GCTStream with embed_dim=1024（保持原始 downstream 结构）
  2. 替换 patch_embed 为 DINOv2 ViT-B/S backbone + adapter 到 1024 维
  3. 加载 lingbot-map 原始 downstream 权重（跳过原始 ViT-L patch_embed）
  4. 冻结 original downstream，只训练 ViT-B/S backbone + adapter
  5. Loss: depth_loss + pose_weight * abs_pose_loss + rel_pose_weight * rel_pose_loss

关键差异 vs train_replica_gct.py:
  - downstream embed_dim 仍为 1024，和 lingbot-map.pt 对齐
  - backbone 替换为 ViT-B/S，并通过 adapter 投影到 1024
  - original downstream 冻结，backbone + adapter 可训练
  - DataLoader / loss / scheduler 设置与 train_replica_gct.py 的 Stage1 流程对齐

Usage:
  # ViT-B backbone training
  python try_train/train_backbone_only.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --checkpoint /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --dinov2_pretrained pretrained/dinov2_vitb14_reg4_pretrain.pth \
    --output_dir try_train/checkpoints/backbone_vitb_5k

  # Smoke test (50 iterations)
  python try_train/train_backbone_only.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --checkpoint /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --dinov2_pretrained pretrained/dinov2_vitb14_reg4_pretrain.pth \
    --output_dir try_train/checkpoints/backbone_vitb_smoke \
    --total_iterations 50 --log_every 5

  # ViT-S fallback (轻量化)
  python try_train/train_backbone_only.py \
    --backbone_type vits \
    --dinov2_pretrained pretrained/dinov2_vits14_reg4_pretrain.pth \
    ...
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
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
# Model creation & freezing
# ---------------------------------------------------------------------------

BACKBONE_CONFIGS = {
    'vitb': {'backbone_dim': 768, 'patch_embed': 'dinov2_vitb14_reg'},
    'vits': {'backbone_dim': 384, 'patch_embed': 'dinov2_vits14_reg'},
    'vitl': {'backbone_dim': 1024, 'patch_embed': 'dinov2_vitl14_reg'},
}

DOWNSTREAM_DIM = 1024


class BackboneAdapter(nn.Module):
    """DINOv2 backbone followed by a projection to the original GCT downstream dim."""

    def __init__(self, backbone, in_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim == out_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
            nn.init.xavier_uniform_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, *args, **kwargs):
        output = self.backbone(*args, **kwargs)
        if isinstance(output, dict):
            output = dict(output)
            output["x_norm_patchtokens"] = self.adapter(output["x_norm_patchtokens"])
            return output
        return self.adapter(output)


def _load_state_dict_file(checkpoint_path, map_location='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return ckpt.get("model", ckpt.get("full_model_state_dict", ckpt.get("model_state_dict", ckpt)))


def _load_dinov2_backbone_weights(backbone, dinov2_pretrained):
    """Load DINOv2 weights into the replacement backbone only."""
    ckpt = torch.load(dinov2_pretrained, map_location='cpu', weights_only=False)
    ckpt = dict(ckpt)
    ckpt.pop('pos_embed', None)
    missing, unexpected = backbone.load_state_dict(ckpt, strict=False)
    if hasattr(backbone, "mask_token"):
        backbone.mask_token.requires_grad_(False)
    print(f"[load_dinov2] Loaded from {dinov2_pretrained}")
    print(f"[load_dinov2] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def _load_original_downstream(model, checkpoint_path):
    """Load original lingbot-map weights, skipping the old ViT-L patch_embed."""
    state_dict = _load_state_dict_file(checkpoint_path, map_location='cpu')
    downstream_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('aggregator.patch_embed')
    }
    missing, unexpected = model.load_state_dict(downstream_state_dict, strict=False)
    downstream_missing = [k for k in missing if not k.startswith('aggregator.patch_embed')]
    if downstream_missing:
        print(f"[load_downstream] Missing downstream keys ({len(downstream_missing)}): {downstream_missing[:5]}...")
    if unexpected:
        print(f"[load_downstream] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print(f"[load_downstream] Loaded original downstream from {checkpoint_path}")


def create_gct_backbone_model(backbone_type, dinov2_pretrained, checkpoint_path, device, use_sdpa=True):
    """Create hybrid GCTStream: ViT-B/S backbone + adapter + original ViT-L downstream."""
    config = BACKBONE_CONFIGS[backbone_type]
    backbone_dim = config['backbone_dim']
    patch_embed_name = config['patch_embed']

    print(f"[create_model] Replacement backbone: {patch_embed_name} (dim={backbone_dim})")
    print(f"[create_model] Downstream dim: {DOWNSTREAM_DIM}")
    print(f"[create_model] Original downstream checkpoint: {checkpoint_path}")
    print(f"[create_model] DINOv2 pretrained: {dinov2_pretrained}")

    model = GCTStream(
        img_size=518,
        patch_size=14,
        embed_dim=DOWNSTREAM_DIM,
        patch_embed=patch_embed_name,
        pretrained_path='',  # load ViT-B/S manually; avoid initializing 1024-dim blocks from 768-dim DINO
        enable_3d_rope=True,
        max_frame_num=100,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        enable_point=True,
    )

    _load_original_downstream(model, checkpoint_path)
    _load_dinov2_backbone_weights(model.aggregator.patch_embed, dinov2_pretrained)
    model.aggregator.patch_embed = BackboneAdapter(
        model.aggregator.patch_embed,
        in_dim=backbone_dim,
        out_dim=DOWNSTREAM_DIM,
    )
    model = model.to(device)

    backbone_params = sum(p.numel() for p in model.aggregator.patch_embed.parameters())
    other_params = sum(p.numel() for p in model.parameters()) - backbone_params

    adapter_params = 0
    if hasattr(model.aggregator.patch_embed, 'adapter'):
        adapter_params = sum(p.numel() for p in model.aggregator.patch_embed.adapter.parameters())
    print(f"[create_model] Backbone+adapter params: {backbone_params:,} ({backbone_params * 4 / 1024**2:.2f} MiB)")
    print(f"[create_model] Adapter params: {adapter_params:,}")
    print(f"[create_model] Other params: {other_params:,}")

    return model


def freeze_non_backbone(model):
    """Freeze original downstream; keep replacement backbone + adapter trainable."""
    # Freeze aggregator blocks + special tokens
    for param in model.aggregator.frame_blocks.parameters():
        param.requires_grad = False

    for param in model.aggregator.global_blocks.parameters():
        param.requires_grad = False

    for name, param in model.aggregator.named_parameters():
        if name in ('camera_token', 'register_token', 'scale_token'):
            param.requires_grad = False

    # Freeze heads
    for param in model.camera_head.parameters():
        param.requires_grad = False

    for param in model.depth_head.parameters():
        param.requires_grad = False

    if getattr(model, 'point_head', None) is not None:
        for param in model.point_head.parameters():
            param.requires_grad = False

    if getattr(model, 'local_point_head', None) is not None:
        for param in model.local_point_head.parameters():
            param.requires_grad = False

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"[freeze_non_backbone] Trainable (backbone+adapter): {trainable:,} ({trainable/total:.2%})")
    print(f"[freeze_non_backbone] Frozen: {frozen:,} ({frozen/total:.2%})")
    print(f"[freeze_non_backbone] Total: {total:,}")

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[freeze_non_backbone] Trainable params sample: {trainable_names[:8]}")


# ---------------------------------------------------------------------------
# Forward pass: backbone with grad + others with no_grad
# ---------------------------------------------------------------------------

def forward_backbone_only(model, images):
    """
    Backbone-only forward pass.

    Design:
      - Aggregator forward with torch.enable_grad() (backbone trainable)
      - PyTorch automatically skips gradients for frozen blocks
      - Heads forward with torch.enable_grad() (to maintain gradient chain)
      - Loss gradients flow back to backbone through frozen layers

    Args:
        model: GCTStream model (backbone trainable, others frozen)
        images: [B, S, 3, H, W] in [0,1] range

    Returns:
        dict with 'depth', 'depth_conf', 'pose_enc', 'images'
    """
    B, S = images.shape[:2]

    # Aggregator forward (backbone trainable, blocks frozen)
    # Use enable_grad() to allow backbone gradient tracking
    model.clean_kv_cache()
    aggregated_tokens_list, patch_start_idx = model.aggregator(
        images,
        selected_idx=[4, 11, 17, 23],
        num_frame_for_scale=S,
        sliding_window_size=-1,
        num_frame_per_block=S,
    )
    model.clean_kv_cache()

    # Camera head forward (frozen, but maintains gradient chain)
    model.camera_head.clean_kv_cache()
    camera_output = model._predict_camera(
        aggregated_tokens_list,
        causal_inference=False,
        num_frame_per_block=S,
        num_frame_for_scale=S,
    )
    model.camera_head.clean_kv_cache()

    # Depth head forward (frozen, but maintains gradient chain)
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
# Depth alignment
# ---------------------------------------------------------------------------

def align_depth_to_gt(depth_pred, depth_gt, valid_mask_gt):
    """Align predicted depth to GT resolution."""
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
    """Train single iteration (backbone only)."""
    # Set modes: backbone implicitly trainable via enable_grad
    # Frozen components stay in eval mode
    model.aggregator.patch_embed.train()
    model.aggregator.frame_blocks.eval()
    model.aggregator.global_blocks.eval()
    model.camera_head.eval()
    model.depth_head.eval()

    images = batch['images'].to(args.device)
    depths = batch['depths'].to(args.device)
    valid_masks = batch['valid_masks'].to(args.device)
    poses = batch['poses'].to(args.device)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        predictions = forward_backbone_only(model, images)

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
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.gradient_clip_norm
        )
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(iters, loss_history['total'], 'b-', linewidth=0.8)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, loss_history['depth'], 'g-', linewidth=0.8)
    axes[0, 1].set_title('Depth Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, loss_history['abs_pose'], 'r-', linewidth=0.8)
    axes[1, 0].set_title('Absolute Pose Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, loss_history['lr'], 'k-', linewidth=0.8)
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = vis_dir / 'loss_curves.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[loss_curves] Saved to {save_path}")


def save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args, path):
    """Save training checkpoint with the same full-model surface as GCT Stage1."""
    backbone_state_dict = {
        k: v for k, v in model.state_dict().items()
        if k.startswith('aggregator.patch_embed')
    }
    checkpoint = {
        'iteration': iteration,
        'backbone_state_dict': backbone_state_dict,
        'full_model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_dict': loss_dict,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"[save_checkpoint] Saved to {path} (iteration {iteration})")


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def print_memory_usage():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[memory] Allocated: {allocated:.3f} GiB, Reserved: {reserved:.3f} GiB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCTStream Backbone-Only Training")

    # Data
    parser.add_argument('--data_root', type=str, required=True)

    # Model
    parser.add_argument('--backbone_type', type=str, default='vitb',
                        choices=['vitb', 'vits', 'vitl'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Original lingbot-map checkpoint for frozen downstream weights")
    parser.add_argument('--dinov2_pretrained', type=str, required=True,
                        help="DINOv2 ViT-B/S pretrained weights path")
    parser.add_argument('--use_sdpa', action='store_true', default=True)
    parser.add_argument('--use_flashinfer', dest='use_sdpa', action='store_false')

    # View sampling
    parser.add_argument('--min_views', type=int, default=2)
    parser.add_argument('--max_views', type=int, default=24)
    parser.add_argument('--sampler_type', type=str, default='spatial_nearby',
                        choices=['temporal_nearby', 'spatial_nearby'])
    parser.add_argument('--spatial_radius', type=float, default=5.0)

    # Training params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iterations', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Learning rate (smaller for backbone)")
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--pose_weight', type=float, default=0.1)
    parser.add_argument('--rel_pose_weight', type=float, default=0.05)
    parser.add_argument('--rel_pose_start_iter', type=int, default=1000)

    # LR scheduler
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-8)

    # Geometric augmentation
    parser.add_argument('--no_geometric_aug', action='store_true')
    parser.add_argument('--no_co_jitter', action='store_true')

    # Output
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--output_dir', type=str,
                        default='./try_train/checkpoints/backbone_vitb')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)

    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(exist_ok=True)

    warmup_iterations = int(args.total_iterations * args.warmup_ratio)

    print(f"\n{'='*60}")
    print(f"GCTStream Backbone-Only Training")
    print(f"{'='*60}")
    print(f"  - Model: Hybrid GCTStream ({args.backbone_type}->1024 adapter), original downstream frozen")
    print(f"  - Original checkpoint: {args.checkpoint}")
    print(f"  - DINOv2 pretrained: {args.dinov2_pretrained}")
    print(f"  - backbone_dim: {BACKBONE_CONFIGS[args.backbone_type]['backbone_dim']} ({args.backbone_type})")
    print(f"  - downstream_dim: {DOWNSTREAM_DIM}")
    print(f"  - Resolution: 518 (ViT-{args.backbone_type[-1].upper()}/14 -> original GCT)")
    print(f"  - lr: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"  - total_iterations: {args.total_iterations}")
    print(f"  - warmup: {warmup_iterations} ({args.warmup_ratio*100:.0f}%)")
    print(f"  - views: [{args.min_views}, {args.max_views}] ({args.sampler_type})")
    print(f"  - geometric_aug: {not args.no_geometric_aug}")
    print(f"  - use_sdpa: {args.use_sdpa}")
    print(f"  - use_amp: {args.use_amp}")
    print(f"  - pose_weight: {args.pose_weight}")
    print(f"  - rel_pose_weight: {args.rel_pose_weight} (start_iter: {args.rel_pose_start_iter})")
    print(f"Data: {args.data_root}")

    # Model
    model = create_gct_backbone_model(
        args.backbone_type,
        args.dinov2_pretrained,
        args.checkpoint,
        args.device,
        use_sdpa=args.use_sdpa,
    )
    freeze_non_backbone(model)

    # Resume
    start_iteration = 0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(resume_ckpt['full_model_state_dict'], strict=False)
        freeze_non_backbone(model)
        start_iteration = resume_ckpt['iteration'] + 1
        print(f"[resume] Loaded from {args.resume}, starting from iteration {start_iteration}")

    # Data
    dataloader = create_replica_dataloader(
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
            dataloader.dataset,
            range(min(args.max_samples, len(dataloader.dataset)))
        )
        dataloader = torch.utils.data.DataLoader(
            limited_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        print(f"[limit] Using {len(limited_dataset)} samples for quick test")

    # Loss
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss() if args.pose_weight > 0 else None
    rel_pose_loss_fn = RelativePoseLoss() if args.rel_pose_weight > 0 else None

    # Optimizer (only backbone parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[Optimizer] Trainable parameters: {trainable_count:,}")

    # LR scheduler
    warmup_iters = int(args.total_iterations * args.warmup_ratio)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.min_lr / args.lr,
        end_factor=1.0,
        total_iters=warmup_iters
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.total_iterations - warmup_iters,
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iters]
    )

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if resume_ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])

    # AMP
    scaler = GradScaler() if args.use_amp else None

    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")

    loss_history = {
        'iterations': [], 'total': [], 'depth': [], 'abs_pose': [], 'rel_pose': [], 'lr': []
    }

    start_time = time.time()
    dataloader_iter = iter(dataloader)
    iteration = start_iteration

    while iteration < args.total_iterations:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        loss_dict = train_one_iteration(
            model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
            iteration, args, scaler
        )

        iteration += 1

        # Log
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

        # Save checkpoint
        if iteration % args.save_every == 0:
            save_path = output_dir / f"checkpoint_iter_{iteration}.pt"
            save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args, save_path)

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
