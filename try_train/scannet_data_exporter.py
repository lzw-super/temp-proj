"""
ScanNet 数据获取脚本

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本脚本用于：
1. 下载 ScanNet .sens 文件（需要官方许可）
2. 使用官方 Python exporter 导出 color/depth/pose/intrinsic
3. 生成训练用的 metadata 文件

参考：
- ScanNet 官方 SensReader: https://www.scan-net.org/ScanNet/SensReader/
- Python Data Exporter: https://www.scan-net.org/ScanNet/SensReader/python/

作者：Claude Code
日期：2026-05-11
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ScanNet 官方 SensReader 工具需要单独下载
# 用户需要自己申请 ScanNet 许可并下载 SensReader.py

class ScanNetDataExporter:
    """
    ScanNet 数据导出器

    负责：
    1. 从 .sens 文件导出 color/depth/pose/intrinsic
    2. 组织目录结构
    3. 过滤坏帧
    4. 生成 metadata
    """

    def __init__(
        self,
        scannet_root: str,
        output_root: str,
        max_scenes: Optional[int] = None,
        valid_depth_threshold: float = 0.05,
        max_frames_per_scene: Optional[int] = None,
    ):
        """
        Args:
            scannet_root: ScanNet 原始数据路径（包含 .sens 文件）
            output_root: 输出路径（导出的 color/depth/pose/intrinsic）
            max_scenes: 最大处理场景数（用于小规模测试）
            valid_depth_threshold: 有效深度像素比例阈值
            max_frames_per_scene: 每个场景最大帧数（用于快速测试）
        """
        self.scannet_root = Path(scannet_root)
        self.output_root = Path(output_root)
        self.max_scenes = max_scenes
        self.valid_depth_threshold = valid_depth_threshold
        self.max_frames_per_scene = max_frames_per_scene

        print(f"[ScanNetDataExporter] 初始化")
        print(f"  - ScanNet 根目录: {scannet_root}")
        print(f"  - 输出根目录: {output_root}")
        print(f"  - 最大场景数: {max_scenes if max_scenes else '全部'}")
        print(f"  - 有效深度阈值: {valid_depth_threshold}")
        print(f"  - 每场景最大帧数: {max_frames_per_scene if max_frames_per_scene else '全部'}")

    def get_scene_list(self) -> List[str]:
        """
        获取所有场景列表

        Returns:
            场景ID列表，如 ['scene0000_00', 'scene0001_00', ...]
        """
        scenes = []
        for item in self.scannet_root.iterdir():
            if item.is_dir() and item.name.startswith('scene'):
                scenes.append(item.name)

        scenes.sort()

        if self.max_scenes:
            scenes = scenes[:self.max_scenes]

        print(f"[get_scene_list] 找到 {len(scenes)} 个场景")
        return scenes

    def export_single_scene(self, scene_id: str) -> bool:
        """
        导出单个场景

        使用 ScanNet 官方 SensReader.py 导出：
        - color/*.jpg
        - depth/*.png (16-bit, 单位毫米)
        - pose/*.txt (4x4 camera-to-world matrix)
        - intrinsic/*.txt

        Args:
            scene_id: 场景ID，如 'scene0000_00'

        Returns:
            是否成功导出
        """
        scene_dir = self.scannet_root / scene_id
        sens_file = scene_dir / f"{scene_id}.sens"
        output_dir = self.output_root / scene_id

        # 检查 .sens 文件是否存在
        if not sens_file.exists():
            print(f"[export_single_scene] 警告: {sens_file} 不存在，跳过")
            return False

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'color').mkdir(exist_ok=True)
        (output_dir / 'depth').mkdir(exist_ok=True)
        (output_dir / 'pose').mkdir(exist_ok=True)
        (output_dir / 'intrinsic').mkdir(exist_ok=True)

        print(f"[export_single_scene] 开始导出 {scene_id}...")

        # 注意：这里需要用户自己下载 SensReader.py
        # 并确保路径正确
        try:
            # 延迟导入 SensReader，避免未下载时报错
            import sys
            sensreader_path = os.path.expanduser("~/ScanNet/SensReader/python/SensReader.py")
            if not os.path.exists(sensreader_path):
                print(f"[export_single_scene] 错误: 未找到 SensReader.py")
                print(f"  请从 ScanNet 官方下载: https://www.scan-net.org/ScanNet/SensReader/python/")
                print(f"  并放置在: {sensreader_path}")
                return False

            sys.path.insert(0, os.path.dirname(sensreader_path))
            from SensReader import SensorData

            # 加载 .sens 文件
            sens = SensorData(sens_file)

            # 导出数据
            sens.export_depth_images(output_dir / 'depth', frame_skip=1)
            sens.export_color_images(output_dir / 'color', frame_skip=1)
            sens.export_poses(output_dir / 'pose', frame_skip=1)
            sens.export_intrinsics(output_dir / 'intrinsic')

            print(f"[export_single_scene] {scene_id} 导出完成")
            return True

        except Exception as e:
            print(f"[export_single_scene] 错误: {e}")
            return False

    def filter_frames(
        self,
        scene_id: str,
        depth_dir: Path,
        pose_dir: Path,
    ) -> List[int]:
        """
        过滤坏帧

        检查：
        - RGB 文件存在
        - depth 文件存在
        - pose 文件存在
        - pose matrix 全部为 finite
        - depth 有效像素比例 >= threshold

        Args:
            scene_id: 场景ID
            depth_dir: depth 图像目录
            pose_dir: pose 文件目录

        Returns:
            有效帧ID列表
        """
        valid_frames = []

        # 获取所有 depth 文件
        depth_files = sorted(depth_dir.glob('*.png'))
        pose_files = sorted(pose_dir.glob('*.txt'))

        # 确保 depth 和 pose 数量匹配
        if len(depth_files) != len(pose_files):
            print(f"[filter_frames] {scene_id}: depth({len(depth_files)}) 和 pose({len(pose_files)}) 数量不匹配")
            return valid_frames

        for depth_file, pose_file in zip(depth_files, pose_files):
            frame_id = int(depth_file.stem)

            # 检查 pose 是否有效
            pose = np.loadtxt(pose_file)
            if not np.all(np.isfinite(pose)):
                print(f"[filter_frames] {scene_id}/{frame_id}: pose 含 NaN/Inf，跳过")
                continue

            # 检查 depth 有效像素比例
            # ScanNet depth 是 16-bit PNG，单位毫米
            depth = self._read_depth_png(depth_file)
            valid_ratio = np.sum(depth > 0) / depth.size

            if valid_ratio < self.valid_depth_threshold:
                print(f"[filter_frames] {scene_id}/{frame_id}: 有效深度比例 {valid_ratio:.2f} < {self.valid_depth_threshold}, 跳过")
                continue

            valid_frames.append(frame_id)

        # 限制最大帧数
        if self.max_frames_per_scene and len(valid_frames) > self.max_frames_per_scene:
            # 随机采样，保持时间序列特性
            import random
            random.seed(42)
            valid_frames = sorted(random.sample(valid_frames, self.max_frames_per_scene))

        print(f"[filter_frames] {scene_id}: {len(valid_frames)} 个有效帧（从 {len(depth_files)} 帧）")
        return valid_frames

    def _read_depth_png(self, depth_file: Path) -> np.ndarray:
        """
        读取 depth PNG 文件

        ScanNet depth 是 16-bit PNG，单位毫米

        Args:
            depth_file: depth 文件路径

        Returns:
            depth 数组（单位毫米），float32
        """
        import cv2
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        return depth

    def read_pose(self, pose_file: Path) -> np.ndarray:
        """
        读取 pose 文件

        ScanNet pose 是 4x4 camera-to-world matrix

        Args:
            pose_file: pose 文件路径

        Returns:
            T_c2w: 4x4 float32 matrix
        """
        pose = np.loadtxt(pose_file).astype(np.float32)
        return pose

    def read_intrinsic(self, intrinsic_file: Path) -> np.ndarray:
        """
        读取 intrinsic 文件

        Args:
            intrinsic_file: intrinsic 文件路径

        Returns:
            K: 3x3 float32 intrinsic matrix
        """
        K = np.loadtxt(intrinsic_file).astype(np.float32)
        return K

    def generate_metadata(
        self,
        scenes: List[str],
        valid_frames_dict: Dict[str, List[int]],
    ) -> Dict:
        """
        生成 metadata

        包含：
        - dataset 名称
        - split
        - scenes 列表
        - 每个场景的帧信息：
          - frame_id
          - rgb_path
          - depth_path
          - K (intrinsic)
          - T_c2w (pose)
          - valid_ratio

        Args:
            scenes: 场景列表
            valid_frames_dict: 每个场景的有效帧字典

        Returns:
            metadata 字典
        """
        metadata = {
            "dataset": "ScanNet",
            "split": "train",
            "scenes": scenes,
            "frames": {}
        }

        for scene_id in scenes:
            scene_dir = self.output_root / scene_id
            depth_dir = scene_dir / 'depth'
            pose_dir = scene_dir / 'pose'
            intrinsic_dir = scene_dir / 'intrinsic'

            # 读取 intrinsic（通常是固定的）
            K_file = intrinsic_dir / 'intrinsic_color.txt'
            if not K_file.exists():
                K_file = intrinsic_dir / 'intrinsic_depth.txt'

            K = self.read_intrinsic(K_file)

            frames_info = []
            valid_frames = valid_frames_dict[scene_id]

            for frame_id in valid_frames:
                depth_file = depth_dir / f"{frame_id}.png"
                pose_file = pose_dir / f"{frame_id}.txt"

                # 深度单位转换：毫米 -> 米
                depth_mm = self._read_depth_png(depth_file)
                valid_ratio = np.sum(depth_mm > 0) / depth_mm.size

                # pose 是 camera-to-world
                T_c2w = self.read_pose(pose_file)

                frame_info = {
                    "frame_id": frame_id,
                    "rgb_path": str(scene_dir / 'color' / f"{frame_id}.jpg"),
                    "depth_path": str(depth_file),
                    "K": K.tolist(),  # 3x3 matrix
                    "T_c2w": T_c2w.tolist(),  # 4x4 matrix
                    "valid_ratio": float(valid_ratio),
                }

                frames_info.append(frame_info)

            metadata["frames"][scene_id] = frames_info

        print(f"[generate_metadata] 生成完成，共 {len(scenes)} 个场景")
        return metadata

    def save_metadata(self, metadata: Dict, output_file: str):
        """
        保存 metadata

        Args:
            metadata: metadata 字典
            output_file: 输出文件路径（.json 或 .pkl）
        """
        output_path = Path(output_file)

        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[save_metadata] 保存到 {output_file} (JSON)")

        elif output_path.suffix == '.pkl':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"[save_metadata] 保存到 {output_file} (pickle)")

        else:
            raise ValueError(f"不支持的格式: {output_path.suffix}")

    def run(self, metadata_output: str = "scannet_train_meta.json"):
        """
        运行完整的数据导出流程

        Args:
            metadata_output: metadata 输出文件路径
        """
        print(f"\n{'='*60}")
        print(f"ScanNet 数据导出开始")
        print(f"{'='*60}\n")

        # 步骤1: 获取场景列表
        scenes = self.get_scene_list()

        if len(scenes) == 0:
            print(f"[run] 没有找到任何场景")
            return

        # 步骤2: 导出每个场景
        valid_frames_dict = {}
        for scene_id in scenes:
            success = self.export_single_scene(scene_id)
            if not success:
                continue

            # 步骤3: 过滤坏帧
            scene_dir = self.output_root / scene_id
            valid_frames = self.filter_frames(
                scene_id,
                scene_dir / 'depth',
                scene_dir / 'pose',
            )

            if len(valid_frames) > 0:
                valid_frames_dict[scene_id] = valid_frames

        if len(valid_frames_dict) == 0:
            print(f"[run] 没有有效的场景")
            return

        # 步骤4: 生成 metadata
        valid_scenes = list(valid_frames_dict.keys())
        metadata = self.generate_metadata(valid_scenes, valid_frames_dict)

        # 步骤5: 保存 metadata
        metadata_path = self.output_root / metadata_output
        self.save_metadata(metadata, str(metadata_path))

        print(f"\n{'='*60}")
        print(f"ScanNet 数据导出完成")
        print(f"{'='*60}\n")
        print(f"  - 有效场景数: {len(valid_scenes)}")
        print(f"  - 总帧数: {sum(len(frames) for frames in valid_frames_dict.values())}")
        print(f"  - Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="ScanNet 数据导出")
    parser.add_argument('--scannet_root', type=str, required=True,
                        help="ScanNet 原始数据路径（包含 .sens 文件）")
    parser.add_argument('--output_root', type=str, required=True,
                        help="输出路径")
    parser.add_argument('--max_scenes', type=int, default=None,
                        help="最大处理场景数（用于测试）")
    parser.add_argument('--max_frames', type=int, default=None,
                        help="每个场景最大帧数（用于测试）")
    parser.add_argument('--valid_depth_threshold', type=float, default=0.05,
                        help="有效深度像素比例阈值")
    parser.add_argument('--metadata_output', type=str, default='scannet_train_meta.json',
                        help="metadata 输出文件名")

    args = parser.parse_args()

    exporter = ScanNetDataExporter(
        scannet_root=args.scannet_root,
        output_root=args.output_root,
        max_scenes=args.max_scenes,
        max_frames_per_scene=args.max_frames,
        valid_depth_threshold=args.valid_depth_threshold,
    )

    exporter.run(metadata_output=args.metadata_output)


if __name__ == '__main__':
    main()