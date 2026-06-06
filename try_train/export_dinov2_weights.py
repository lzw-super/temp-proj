"""
导出DINOv2 ViT-B/S权重供LingBot使用

Usage:
  python try_train/export_dinov2_weights.py --model vitb --output pretrained/dinov2_vitb14_reg4_pretrain.pth
  python try_train/export_dinov2_weights.py --model vits --output pretrained/dinov2_vits14_reg4_pretrain.pth
"""
import torch
import argparse
from pathlib import Path

DINOV2_REPO = "/home/lizhengwu/desktop/temp_proj/dinov2"


def export_weights(model_name: str, output_path: str):
    """导出DINOv2权重

    Args:
        model_name: 模型类型 ('vitb', 'vits', 'vitl')
        output_path: 输出路径
    """
    hub_name = f"dinov2_{model_name}14_reg"

    print(f"[export] Loading {hub_name} from {DINOV2_REPO}...")

    model = torch.hub.load(
        DINOV2_REPO,
        hub_name,
        source="local",
        pretrained=True,
    )

    state_dict = model.state_dict()
    torch.save(state_dict, output_path)

    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 ** 2)

    print(f"[export] ✓ {hub_name} weights saved to {output_path}")
    print(f"[export]   Parameters: {params:,} ({size_mb:.2f} MiB FP32)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出DINOv2预训练权重")
    parser.add_argument(
        "--model",
        choices=["vitb", "vits", "vitl"],
        default="vitb",
        help="模型类型: vitb=ViT-B, vits=ViT-S, vitl=ViT-L"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出权重文件路径 (.pth)"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_weights(args.model, args.output)