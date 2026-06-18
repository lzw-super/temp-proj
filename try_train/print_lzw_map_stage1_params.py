"""Build LZW-Map Stage1 and print parameter counts.

Usage:
  python try_train/print_lzw_map_stage1_params.py
  python try_train/print_lzw_map_stage1_params.py --no_pretrained
  python try_train/print_lzw_map_stage1_params.py --enable_point
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lingbot_map.models.lzw_map import (  # noqa: E402
    create_lzw_map_stage1,
    format_parameter_rows,
    lzw_map_parameter_summary,
)


DEFAULT_PRETRAINED = PROJECT_ROOT / "pretrained" / "dinov2_vitb14_reg4_pretrain.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Print LZW-Map Stage1 parameter counts.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=str(DEFAULT_PRETRAINED) if DEFAULT_PRETRAINED.exists() else "",
        help="DINOv2 ViT-B/14 reg checkpoint used to initialize the student backbone and blocks.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Do not load DINOv2 pretrained weights; only instantiate the architecture.",
    )
    parser.add_argument(
        "--enable_point",
        action="store_true",
        help="Also build the original world-point DPT head. Stage1 depth+pose training leaves this disabled by default.",
    )
    parser.add_argument(
        "--use_flashinfer",
        action="store_true",
        help="Use FlashInfer blocks instead of the SDPA fallback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pretrained_path = "" if args.no_pretrained else args.pretrained_path

    print("[lzw-map-stage1] building model...")
    print(f"  pretrained_path: {pretrained_path or '(none)'}")
    print(f"  point_head: {'enabled' if args.enable_point else 'disabled'}")
    print(f"  backend: {'FlashInfer' if args.use_flashinfer else 'SDPA'}")

    model = create_lzw_map_stage1(
        pretrained_path=pretrained_path,
        enable_point=args.enable_point,
        use_sdpa=not args.use_flashinfer,
    )

    print("\n[lzw-map-stage1] architecture")
    print(f"  patch_embed: {model.patch_embed}")
    print(f"  embed_dim: {model.embed_dim}")
    print(f"  aggregator_depth: {model.aggregator_depth}")
    print(f"  selected_idx: {model.selected_idx}")
    print(f"  camera_trunk_depth: {model.camera_trunk_depth}")
    print(f"  camera_num_heads: {model.camera_num_heads}")
    print(f"  depth_head: unchanged DPTHead")

    print("\n[lzw-map-stage1] parameters")
    rows = lzw_map_parameter_summary(model)
    for line in format_parameter_rows(rows):
        print(line)


if __name__ == "__main__":
    main()
