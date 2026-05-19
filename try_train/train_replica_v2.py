"""
Head-Only 训练脚本（论文4.1对齐版）

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本脚本实现完整的训练流程：
1. 加载 Replica 数据
2. 构建 head-only 模型（冻结 backbone）
3. 训练循环：depth loss + absolute pose loss + relative pose loss
4. 参数：AdamW, lr=2e-4, wd=0.05, warmup+cosine scheduler
5. Checkpoint 保存

遵循论文 Sec. 4.1 的第一阶段设置。

作者：Claude Code
日期：2026-05-12
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))

from replica_dataset import create_replica_dataloader
from head_only_model import create_head_only_model, DepthLoss, PoseLoss, RelativePoseLoss


def train_one_iteration(
    model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
    iteration, args, scaler=None
):
    """训练单个 iteration"""
    model.train()

    images = batch['images'].to(args.device)
    depths = batch['depths'].to(args.device)
    valid_masks = batch['valid_masks'].to(args.device)
    poses = batch['poses'].to(args.device)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        predictions = model(images)

        # Depth loss
        depth_loss = depth_loss_fn(predictions['depth'], depths, valid_masks)
        loss = depth_loss
        loss_dict = {'depth': depth_loss.item()}

        # Absolute pose loss
        if pose_loss_fn is not None and 'pose_enc' in predictions:
            pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
            loss = loss + args.pose_weight * pose_loss
            loss_dict['abs_pose'] = pose_loss.item()

        # Relative pose loss（延迟开启）
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

    # 更新 scheduler
    if scheduler is not None:
        scheduler.step()

    loss_dict['total'] = loss.item()

    return loss_dict


def save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args, path):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_dict': loss_dict,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"[save_checkpoint] Saved to {path} (iteration {iteration})")


def main():
    parser = argparse.ArgumentParser(description="Head-Only Training (论文4.1对齐版)")

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help="Replica data root directory")

    # 模型参数
    parser.add_argument('--backbone', type=str, default='dinov2_vits14')
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--train_pose', type=bool, default=True)  # 默认启用
    parser.add_argument('--rel_pose', type=bool, default=True)   # 默认启用

    # 视角采样参数（论文4.1：views∈[2,24]）
    parser.add_argument('--min_views', type=int, default=2,
                        help="最小视角数（论文=2）")
    parser.add_argument('--max_views', type=int, default=2,
                        help="最大视角数（论文=24，当前head-only建议≤8避免显存问题）")
    parser.add_argument('--sampler_type', type=str, default='temporal_nearby',
                        choices=['temporal_nearby', 'spatial_nearby'],
                        help="采样策略（默认temporal_nearby）")
    parser.add_argument('--spatial_radius', type=float, default=5.0,
                        help="spatial nearby 3D距离阈值（米）")

    # 训练参数（论文4.1设置）
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iterations', type=int, default=5000,
                        help="总训练 iterations（论文160K，资源不足可减少）")
    parser.add_argument('--lr', type=float, default=2e-4,          # 论文设置
                        help="Base learning rate (论文=2e-4)")
    parser.add_argument('--weight_decay', type=float, default=0.05, # 论文设置
                        help="Weight decay (论文=0.05)")
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # Loss 权重
    parser.add_argument('--pose_weight', type=float, default=0.1,
                        help="Absolute pose loss weight (论文=0.1)")
    parser.add_argument('--rel_pose_weight', type=float, default=0.05,
                        help="Relative pose loss weight (论文=0.05)")
    parser.add_argument('--rel_pose_start_iter', type=int, default=1000,
                        help="Relative pose loss 延迟开启的 iteration")

    # Scheduler 参数（论文4.1设置）
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help="Warmup iterations ratio (论文=5%)")
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help="Minimum learning rate (论文=1e-8)")

    # Geometric augmentation 参数
    parser.add_argument('--no_geometric_aug', action='store_true',
                        help="禁用几何增强（spatial rescale + aspect-ratio）")
    parser.add_argument('--no_co_jitter', action='store_true',
                        help="禁用 co-jitter（多view共享颜色扰动改为独立jitter）")

    # 其他参数
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=1000,
                        help="每 N iterations 保存 checkpoint")
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None,
                        help="限制样本数量，用于快速测试")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算 warmup iterations
    warmup_iterations = int(args.total_iterations * args.warmup_ratio)

    print(f"\n{'='*60}")
    print(f"Head-Only Training (论文4.1对齐版)")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  - backbone: {args.backbone}")
    print(f"  - lr: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"  - total_iterations: {args.total_iterations}")
    print(f"  - warmup: {warmup_iterations} ({args.warmup_ratio*100:.0f}%)")
    print(f"  - train_pose: {args.train_pose}")
    print(f"  - rel_pose_weight: {args.rel_pose_weight} (start at {args.rel_pose_start_iter})")
    print(f"  - views: [{args.min_views}, {args.max_views}]")
    print(f"  - geometric_aug: {not args.no_geometric_aug}")
    print(f"  - co_jitter: {not args.no_co_jitter}")
    print(f"Data: {args.data_root}")

    # 1. DataLoader（支持视角范围采样）
    if args.min_views != args.max_views:
        print(f"  - Views range: [{args.min_views}, {args.max_views}]")
    train_dataloader = create_replica_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        min_views=args.min_views,
        max_views=args.max_views,
        max_dim=args.img_size,
        shuffle=True,
        num_workers=4,
        seed=args.seed,
        sampler_type=args.sampler_type,
        spatial_radius=args.spatial_radius,
        spatial_rescale_range=None if args.no_geometric_aug else (0.8, 1.2),
        aspect_ratio_range=None if args.no_geometric_aug else (0.33, 1.0),
        co_jitter=not args.no_co_jitter,
    )

    # 限制样本数量用于快速测试
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
        print(f"[限制样本] 使用 {len(limited_dataset)} 个样本进行快速测试")

    # 2. Model
    model = create_head_only_model(
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size,
        train_depth_head=True,
        train_pose_head=args.train_pose,
        num_views=args.min_views,
    )
    model = model.to(args.device)

    # 3. Loss
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss() if args.train_pose else None
    rel_pose_loss_fn = RelativePoseLoss() if args.rel_pose and args.train_pose else None

    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"[Optimizer] 可训练参数数量: {len(trainable_params)}")

    # 5. Scheduler (5% warmup + cosine decay)
    # Warmup: 从 min_lr (1e-8) 线性增长到 base lr (2e-4)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.min_lr / args.lr,  # 开始时的 LR factor
        end_factor=1.0,                       # 结束时的 LR factor (base lr)
        total_iters=warmup_iterations
    )

    # Cosine decay: 从 base lr 下降到 min_lr
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

    print(f"[Scheduler] Warmup: {warmup_iterations} iterations, then cosine decay to {args.min_lr}")

    # 6. Scaler
    scaler = GradScaler() if args.use_amp else None

    # 7. Resume
    start_iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        print(f"[Resume] 从 iteration {start_iteration} 继续")

    # 8. Training loop (iteration-based)
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")

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

        # Logging
        if iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Iter {iteration}/{args.total_iterations}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(depth: {loss_dict.get('depth', 0):.4f}, "
                  f"abs_pose: {loss_dict.get('abs_pose', 0):.4f}, "
                  f"rel_pose: {loss_dict.get('rel_pose', 0):.4f}) "
                  f"LR: {current_lr:.2e} "
                  f"Time: {elapsed:.2f}s")

        # Save checkpoint
        if iteration % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                           output_dir / f'checkpoint_iter_{iteration}.pt')

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, iteration, loss_dict, args,
                   output_dir / 'checkpoint_final.pt')

    print(f"\n{'='*60}")
    print(f"Training completed")
    print(f"{'='*60}")
    print(f"  - Total iterations: {iteration}")
    print(f"  - Final loss: {loss_dict['total']:.4f}")
    print(f"  - Checkpoint saved to: {output_dir}")


if __name__ == '__main__':
    main()