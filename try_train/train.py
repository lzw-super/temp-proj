"""
Head-Only 训练脚本

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本脚本实现完整的训练流程：
1. 加载 ScanNet 数据
2. 构建 head-only 模型（冻结 backbone）
3. 训练循环：depth loss + pose loss
4. 参数：AdamW, lr=1e-3, gradient clipping
5. Checkpoint 保存

作者：Claude Code
日期：2026-05-11
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent))

from scannet_dataset import create_dataloader
from head_only_model import create_head_only_model, DepthLoss, PoseLoss


def train_one_epoch(model, dataloader, optimizer, depth_loss_fn, pose_loss_fn, epoch, args, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(args.device)
        depths = batch['depths'].to(args.device)
        valid_masks = batch['valid_masks'].to(args.device)
        poses = batch['poses'].to(args.device)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            predictions = model(images)

            if 'depth' in predictions:
                depth_loss = depth_loss_fn(predictions['depth'], depths, valid_masks)
                loss = depth_loss

            if 'pose_enc' in predictions and pose_loss_fn is not None:
                pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
                loss = loss + args.pose_weight * pose_loss

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

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % args.log_every == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} Loss: {total_loss/num_batches:.4f}")

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, args, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"[save_checkpoint] Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Head-Only Training")

    parser.add_argument('--train_metadata', type=str, required=True)
    parser.add_argument('--val_metadata', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14')
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--train_pose', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--pose_weight', type=float, default=0.1)
    parser.add_argument('--use_amp', type=bool, default=True)

    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
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

    print(f"\n{'='*60}")
    print(f"Head-Only Training")
    print(f"{'='*60}")
    print(f"Config: backbone={args.backbone}, lr={args.lr}, epochs={args.num_epochs}")

    # 1. DataLoader
    train_dataloader = create_dataloader(
        metadata_path=args.train_metadata,
        batch_size=args.batch_size,
        num_views=args.num_views,
        max_dim=args.img_size,
        shuffle=True,
        num_workers=4,
    )

    # 2. Model
    model = create_head_only_model(
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size,
        train_depth_head=True,
        train_pose_head=args.train_pose,
        num_views=args.num_views,
    )
    model = model.to(args.device)

    # 3. Loss
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss() if args.train_pose else None

    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 5. Scaler
    scaler = GradScaler() if args.use_amp else None

    # 6. Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 7. Training loop
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, depth_loss_fn, pose_loss_fn, epoch, args, scaler)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args, output_dir / f'checkpoint_epoch_{epoch}.pt')

    print(f"\n{'='*60}")
    print(f"Training completed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()