"""LZW-Map lightweight Stage1 model builders and parameter summaries."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import torch.nn as nn

from lingbot_map.models.gct_stream import GCTStream


LZW_MAP_STAGE1_SELECTED_IDX = [2, 5, 8, 11]


class LZWMapStage1(GCTStream):
    """Stage1-ready lightweight LZW-Map model.

    Architecture:
    - DINOv2 ViT-B/14 reg backbone, embed_dim=768
    - Aggregator frame/global blocks: 12 groups
    - Camera pose head trunk: 2 blocks
    - Depth head unchanged from the original DPTHead shape
    """

    model_name = "lzw-map-stage1"

    def __init__(
        self,
        pretrained_path: str = "",
        enable_point: bool = False,
        use_sdpa: bool = True,
        **kwargs,
    ) -> None:
        defaults = dict(
            img_size=518,
            patch_size=14,
            embed_dim=768,
            patch_embed="dinov2_vitb14_reg",
            pretrained_path=pretrained_path or "",
            aggregator_depth=12,
            selected_idx=list(LZW_MAP_STAGE1_SELECTED_IDX),
            camera_trunk_depth=2,
            camera_num_heads=12,
            enable_camera=True,
            enable_point=enable_point,
            enable_local_point=False,
            enable_depth=True,
            enable_track=False,
            enable_3d_rope=True,
            max_frame_num=100,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa,
            use_gradient_checkpoint=True,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


def create_lzw_map_stage1(
    pretrained_path: str = "",
    enable_point: bool = False,
    use_sdpa: bool = True,
    **kwargs,
) -> LZWMapStage1:
    """Create the default lightweight LZW-Map Stage1 model."""
    return LZWMapStage1(
        pretrained_path=pretrained_path,
        enable_point=enable_point,
        use_sdpa=use_sdpa,
        **kwargs,
    )


def _parameter_stats(parameters: Iterable[nn.Parameter]) -> Dict[str, int]:
    params = list(parameters)
    return {
        "total": sum(p.numel() for p in params),
        "trainable": sum(p.numel() for p in params if p.requires_grad),
        "bytes": sum(p.numel() * p.element_size() for p in params),
    }


def _module_stats(module: Optional[nn.Module]) -> Dict[str, int]:
    if module is None:
        return {"total": 0, "trainable": 0, "bytes": 0}
    return _parameter_stats(module.parameters())


def _subtract_stats(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {key: left[key] - right[key] for key in ("total", "trainable", "bytes")}


def _add_stats(*stats: Dict[str, int]) -> Dict[str, int]:
    return {
        key: sum(item[key] for item in stats)
        for key in ("total", "trainable", "bytes")
    }


def lzw_map_parameter_summary(model: nn.Module) -> "OrderedDict[str, Dict[str, int]]":
    """Return non-overlapping high-level parameter groups for an LZW/GCT model."""
    rows: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    rows["total"] = _module_stats(model)

    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None:
        rows["aggregator.total"] = _module_stats(aggregator)
        rows["aggregator.patch_embed_backbone"] = _module_stats(getattr(aggregator, "patch_embed", None))
        rows["aggregator.frame_blocks"] = _module_stats(getattr(aggregator, "frame_blocks", None))
        rows["aggregator.global_blocks"] = _module_stats(getattr(aggregator, "global_blocks", None))

        special_token_names = {"camera_token", "register_token", "scale_token"}
        rows["aggregator.special_tokens"] = _parameter_stats(
            param
            for name, param in aggregator.named_parameters(recurse=False)
            if name in special_token_names
        )
        known_aggregator = _add_stats(
            rows["aggregator.patch_embed_backbone"],
            rows["aggregator.frame_blocks"],
            rows["aggregator.global_blocks"],
            rows["aggregator.special_tokens"],
        )
        rows["aggregator.other"] = _subtract_stats(rows["aggregator.total"], known_aggregator)

    rows["camera_head"] = _module_stats(getattr(model, "camera_head", None))
    rows["depth_head"] = _module_stats(getattr(model, "depth_head", None))
    rows["point_head"] = _module_stats(getattr(model, "point_head", None))
    rows["local_point_head"] = _module_stats(getattr(model, "local_point_head", None))
    return rows


def format_parameter_rows(rows: "OrderedDict[str, Dict[str, int]]") -> List[str]:
    """Format parameter rows as a compact fixed-width table."""
    header = f"{'part':32s} {'params':>16s} {'trainable':>16s} {'size':>12s}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, stats in rows.items():
        lines.append(
            f"{name:32s} "
            f"{stats['total']:16,d} "
            f"{stats['trainable']:16,d} "
            f"{stats['bytes'] / 1024 ** 2:11.2f}M"
        )
    return lines
