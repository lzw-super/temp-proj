"""Validate LZW-Map Stage2 under strict GCA streaming forward."""

from __future__ import annotations

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

from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss  # noqa: E402
from replica_dataset import ReplicaDataset  # noqa: E402
from lingbot_map.models.lzw_map import create_lzw_map_stage2  # noqa: E402
from lingbot_map.utils.load_fn import load_and_preprocess_images  # noqa: E402
from train_aggregator_only import forward_aggregator_heads, load_depth_gct_model  # noqa: E402
from train_lzw_map_stage1 import (  # noqa: E402
    DEFAULT_DINOV2_REPO,
    load_backbone_from_original_dinov2_repo,
)
from train_lzw_map_stage2 import (  # noqa: E402
    forward_lzw_stage2_streaming,
    load_lzw_state,
)
from validate_lzw_map_stage1 import (  # noqa: E402
    align_depth_to_gt,
    compute_pose_metrics,
    collect_gt_visuals,
    maybe_empty_cache,
    median_align_depth,
    plot_metrics,
    plot_pose_metrics,
    save_pose_metrics,
    save_pose_outputs,
    scale_align_pose_translation,
)


def autocast_context(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def build_lzw_stage2_model(
    device,
    dinov2_repo,
    dinov2_hub_name,
    seed,
    use_sdpa,
    max_frame_num,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = create_lzw_map_stage2(
        pretrained_path="",
        enable_point=False,
        use_sdpa=use_sdpa,
        max_frame_num=max_frame_num,
    )
    load_backbone_from_original_dinov2_repo(model, dinov2_repo, dinov2_hub_name)
    return model.to(device).eval()


def load_trained_lzw_stage2_model(
    checkpoint_path,
    device,
    dinov2_repo,
    dinov2_hub_name,
    seed,
    use_sdpa,
    max_frame_num,
):
    model = build_lzw_stage2_model(
        device,
        dinov2_repo,
        dinov2_hub_name,
        seed,
        use_sdpa,
        max_frame_num,
    )
    checkpoint = load_lzw_state(model, checkpoint_path, allow_head_only=False)
    return model.to(device).eval(), checkpoint


def forward_model(model, model_kind, images, sliding_window_size, num_frame_for_scale):
    if model_kind == "lingbot":
        return forward_aggregator_heads(model, images)
    if model_kind == "lzw_stage2":
        return forward_lzw_stage2_streaming(
            model,
            images,
            sliding_window_size=sliding_window_size,
            num_frame_for_scale=num_frame_for_scale,
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def evaluate_model(
    model,
    model_kind,
    dataset,
    data_root,
    device,
    num_samples,
    num_vis,
    label,
    sliding_window_size,
    num_frame_for_scale,
):
    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    official_pose_loss_fn = PoseLoss(quat_convention="xyzw")
    official_rel_pose_loss_fn = RelativePoseLoss(quat_convention="xyzw")
    legacy_pose_loss_fn = PoseLoss(quat_convention="wxyz")
    legacy_rel_pose_loss_fn = RelativePoseLoss(quat_convention="wxyz")
    metrics = {
        "metric_depth_loss": [],
        "scale_aligned_depth_loss": [],
        "official_xyzw_scale_aligned_abs_pose_loss": [],
        "official_xyzw_scale_aligned_rel_pose_loss": [],
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

    model.eval()
    sample_count = min(num_samples, len(dataset))
    for sample_idx in range(sample_count):
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
            predictions = forward_model(
                model,
                model_kind,
                images,
                sliding_window_size=sliding_window_size,
                num_frame_for_scale=num_frame_for_scale,
            )

        depth_pred, masks = align_depth_to_gt(predictions["depth"], depths, masks)
        depth_pred = depth_pred.float().clamp(min=1e-6)
        depths = depths.float()
        aligned_depth = median_align_depth(depth_pred, depths, masks)
        aligned_pose = scale_align_pose_translation(predictions["pose_enc"].float(), poses.float())

        metric_depth_loss = depth_loss_fn(depth_pred, depths, masks)
        aligned_depth_loss = depth_loss_fn(aligned_depth, depths, masks)
        official_abs_pose_loss = official_pose_loss_fn(aligned_pose, poses.float())
        official_rel_pose_loss = official_rel_pose_loss_fn(aligned_pose, poses.float())
        legacy_abs_pose_loss = legacy_pose_loss_fn(aligned_pose, poses.float())
        legacy_rel_pose_loss = legacy_rel_pose_loss_fn(aligned_pose, poses.float())
        pose_metrics = compute_pose_metrics(predictions["pose_enc"].float(), poses.float())

        metrics["metric_depth_loss"].append(float(metric_depth_loss))
        metrics["scale_aligned_depth_loss"].append(float(aligned_depth_loss))
        metrics["official_xyzw_scale_aligned_abs_pose_loss"].append(float(official_abs_pose_loss))
        metrics["official_xyzw_scale_aligned_rel_pose_loss"].append(float(official_rel_pose_loss))
        metrics["legacy_wxyz_scale_aligned_abs_pose_loss"].append(float(legacy_abs_pose_loss))
        metrics["legacy_wxyz_scale_aligned_rel_pose_loss"].append(float(legacy_rel_pose_loss))
        metrics["scale_aligned_abs_pose_loss"].append(float(official_abs_pose_loss))
        metrics["scale_aligned_rel_pose_loss"].append(float(official_rel_pose_loss))
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
            "official_xyzw_scale_aligned_abs_pose_loss": float(official_abs_pose_loss),
            "official_xyzw_scale_aligned_rel_pose_loss": float(official_rel_pose_loss),
            "legacy_wxyz_scale_aligned_abs_pose_loss": float(legacy_abs_pose_loss),
            "legacy_wxyz_scale_aligned_rel_pose_loss": float(legacy_rel_pose_loss),
        })

        print(
            f"  [{label} {sample_idx + 1}/{sample_count}] "
            f"depth={float(metric_depth_loss):.4f} "
            f"depth_aligned={float(aligned_depth_loss):.4f} "
            f"auc3={pose_metrics['pose_xyzw_auc3']:.2f} "
            f"auc30={pose_metrics['pose_xyzw_auc30']:.2f} "
            f"ate={pose_metrics['pose_ate_sim3_rmse_m']:.4f}m "
            f"xyzw_rpeR={pose_metrics['pose_xyzw_rpe_rot_mean_deg']:.2f}deg "
            f"xyzw_rpeT={pose_metrics['pose_xyzw_rpe_trans_rmse_m']:.4f}m "
            f"xyzw_abs={float(official_abs_pose_loss):.4f} "
            f"legacy_abs={float(legacy_abs_pose_loss):.4f}"
        )

    averaged = {key: float(np.mean(values)) for key, values in metrics.items()}
    return averaged, visuals


def _depth_range(gt):
    valid_gt = gt[np.isfinite(gt) & (gt > 0)]
    if not valid_gt.size:
        return 0.0, 1.0
    return float(np.percentile(valid_gt, 2)), float(np.percentile(valid_gt, 98))


def plot_depth_comparison(predictions, gt_visuals, frame_ids, output_path, scale_key):
    model_names = list(predictions)
    columns = [*model_names, "Ground Truth"]
    fig, axes = plt.subplots(
        len(gt_visuals),
        len(columns),
        figsize=(5.0 * len(columns), 4.2 * len(gt_visuals)),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        vmin, vmax = _depth_range(gt)
        row_images = [predictions[name][scale_key][row] for name in model_names] + [gt]
        for col, (name, depth) in enumerate(zip(columns, row_images)):
            if depth.shape != gt.shape:
                depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            image = axes[row, col].imshow(depth, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(
                f"{name} | frame {frame_ids[row]}\nGT scale {vmin:.2f}-{vmax:.2f} m"
            )
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    scale_title = "Metric Scale" if scale_key == "metric" else "Median-Aligned Scale"
    fig.suptitle(f"LZW-Map Stage2 Depth Comparison ({scale_title})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(predictions, gt_visuals, frame_ids, output_path):
    model_names = list(predictions)
    columns = ["Ground Truth", *[f"{name} |Error|" for name in model_names]]
    fig, axes = plt.subplots(
        len(gt_visuals),
        len(columns),
        figsize=(5.0 * len(columns), 4.2 * len(gt_visuals)),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        valid = np.isfinite(gt) & (gt > 0)
        depth_vmin, depth_vmax = _depth_range(gt)
        errors = []
        for name in model_names:
            depth = predictions[name]["aligned"][row]
            if depth.shape != gt.shape:
                depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            error = np.abs(depth - gt)
            error[~valid] = 0
            errors.append(error)
        error_values = np.concatenate([error[valid] for error in errors]) if valid.any() else np.array([])
        error_vmax = float(np.percentile(error_values, 95)) if error_values.size else 1.0

        panels = [gt, *errors]
        for col, (name, panel) in enumerate(zip(columns, panels)):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LZW-Map Stage2 outputs")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--original_model", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--dinov2_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--dinov2_hub_name", default="dinov2_vitb14_reg")
    parser.add_argument("--num_views", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--stage2_sliding_window", type=int, default=4)
    parser.add_argument("--stage2_num_frame_for_scale", type=int, default=8)
    parser.add_argument("--max_frame_num", type=int, default=400)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    seed = int(ckpt_args.get("seed", 42))
    arch = checkpoint.get("architecture", {})
    trained_pose_quat_convention = (
        ckpt_args.get("pose_quat_convention")
        or arch.get("pose_quat_convention")
        or "unknown"
    )
    trained_pose_anchor_scale_norm = (
        ckpt_args.get("use_pose_anchor_scale_norm")
        if "use_pose_anchor_scale_norm" in ckpt_args
        else arch.get("use_pose_anchor_scale_norm", "unknown")
    )
    dinov2_repo = args.dinov2_repo or arch.get("dinov2_repo") or DEFAULT_DINOV2_REPO
    dinov2_hub_name = args.dinov2_hub_name or arch.get("dinov2_hub_name") or "dinov2_vitb14_reg"

    output_dir = Path(args.output_dir or (Path(args.stage2_checkpoint).parent / "vis_validation_stage2"))
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
    print("LZW-Map Stage2 Validation")
    print("=" * 72)
    print(f"  LingBot original: {args.original_model}")
    print(f"  LZW Stage2:       {args.stage2_checkpoint}")
    print(f"  Trained pose q:   {trained_pose_quat_convention}")
    print(f"  Pose anchor norm: {trained_pose_anchor_scale_norm}")
    print(f"  DINOv2 repo:      {dinov2_repo}")
    print(f"  Samples/views:    {args.num_samples}/{args.num_views}")
    print(f"  GCA k / anchors:  {args.stage2_sliding_window}/{args.stage2_num_frame_for_scale}")
    print(f"  Output:           {output_dir}")
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
        args.stage2_sliding_window,
        args.stage2_num_frame_for_scale,
    )
    del model
    gc.collect()
    maybe_empty_cache()

    model = build_lzw_stage2_model(
        args.device,
        dinov2_repo,
        dinov2_hub_name,
        seed,
        args.use_sdpa,
        args.max_frame_num,
    )
    results["LZW Stage2 Init"], predictions["LZW Stage2 Init"] = evaluate_model(
        model,
        "lzw_stage2",
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "LZWStage2Init",
        args.stage2_sliding_window,
        args.stage2_num_frame_for_scale,
    )
    del model
    gc.collect()
    maybe_empty_cache()

    model, loaded_checkpoint = load_trained_lzw_stage2_model(
        args.stage2_checkpoint,
        args.device,
        dinov2_repo,
        dinov2_hub_name,
        seed,
        args.use_sdpa,
        args.max_frame_num,
    )
    results["LZW Stage2 Trained"], predictions["LZW Stage2 Trained"] = evaluate_model(
        model,
        "lzw_stage2",
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "LZWStage2Trained",
        args.stage2_sliding_window,
        args.stage2_num_frame_for_scale,
    )
    del model
    gc.collect()
    maybe_empty_cache()

    payload = {
        "metadata": {
            "original_model": args.original_model,
            "stage2_checkpoint": args.stage2_checkpoint,
            "iteration": loaded_checkpoint.get("iteration", checkpoint.get("iteration")),
            "seed": seed,
            "trained_pose_quat_convention": trained_pose_quat_convention,
            "trained_pose_anchor_scale_norm": trained_pose_anchor_scale_norm,
            "dinov2_repo": dinov2_repo,
            "dinov2_hub_name": dinov2_hub_name,
            "num_samples": args.num_samples,
            "num_views": args.num_views,
            "stage2_sliding_window": args.stage2_sliding_window,
            "stage2_num_frame_for_scale": args.stage2_num_frame_for_scale,
            "stage2_forward": "strict GCA streaming",
            "depth_alignment": "per-view GT median scale",
            "pose_alignment": {
                "ate": "Umeyama Sim(3) alignment of predicted camera centers to GT centers",
                "scale_aligned_loss": "official XYZW PoseLoss/RelativePoseLoss; legacy_wxyz_* is diagnostic only",
                "auc": (
                    "pairwise relative-pose AUC@{3,5,15,30} in percent; C2W poses are converted "
                    "to W2C, aligned to the first camera, then scored by max(rotation angular error, "
                    "translation-direction angular error)"
                ),
                "rpe": "relative pose error over all unordered pairs in the sampled validation clip",
            },
            "pose_quaternion_conventions": {
                "xyzw": "official LingBot pose encoding, scalar-last",
                "wxyz": "legacy local training-loss interpretation, scalar-first",
            },
        },
        "results": results,
    }
    with open(output_dir / "lzw_map_stage2_validation_results.json", "w") as handle:
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

    print("\n" + "-" * 104)
    print(
        f"{'Model':<22} {'Depth':>10} {'DepthA':>10} "
        f"{'AUC@3':>9} {'AUC@30':>9} {'ATE(m)':>10} "
        f"{'XYZW RPE-R':>12} {'XYZW RPE-T':>12}"
    )
    print("-" * 104)
    for name, metrics in results.items():
        print(
            f"{name:<22} "
            f"{metrics['metric_depth_loss']:>10.5f} "
            f"{metrics['scale_aligned_depth_loss']:>10.5f} "
            f"{metrics['pose_xyzw_auc3']:>9.2f} "
            f"{metrics['pose_xyzw_auc30']:>9.2f} "
            f"{metrics['pose_ate_sim3_rmse_m']:>10.5f} "
            f"{metrics['pose_xyzw_rpe_rot_mean_deg']:>12.4f} "
            f"{metrics['pose_xyzw_rpe_trans_rmse_m']:>12.5f}"
        )
    print("-" * 104)

    print("\nOfficial XYZW pose metrics (AUC higher is better; errors lower are better)")
    print("-" * 114)
    print(
        f"{'Model':<22} {'AUC@3':>9} {'AUC@5':>9} {'AUC@15':>9} {'AUC@30':>9} "
        f"{'AbsRot':>10} {'AnchRot':>10} {'RPERot':>10} {'RPETrans':>10} {'Sim3Scale':>10}"
    )
    print("-" * 114)
    for name, metrics in results.items():
        print(
            f"{name:<22} "
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
    print("-" * 114)

    initial = results["LZW Stage2 Init"]
    trained = results["LZW Stage2 Trained"]
    if initial["metric_depth_loss"] > 0:
        print(
            "LZW Stage2 trained vs init metric-depth improvement: "
            f"{(initial['metric_depth_loss'] - trained['metric_depth_loss']) / initial['metric_depth_loss'] * 100:+.1f}%"
        )
    if initial["scale_aligned_depth_loss"] > 0:
        print(
            "LZW Stage2 trained vs init aligned-depth improvement: "
            f"{(initial['scale_aligned_depth_loss'] - trained['scale_aligned_depth_loss']) / initial['scale_aligned_depth_loss'] * 100:+.1f}%"
        )
    print(f"Validation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
