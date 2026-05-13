"""
Foldback Video Sampler for Replica Long Sequence Training

实现论文4.2 Streaming Model Training的长序列采样策略：
1. Foldback sampler：边界反向继续，避免固定前进偏置
2. Progressive view curriculum：views数随iteration线性增加
3. Local window sampling：用于relative pose loss的局部窗口

作者：Claude Code
日期：2026-05-12
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class FoldbackVideoSampler:
    """
    Foldback Video Sampler

    产生连续长序列视频帧，边界反向继续：
    - 从随机起始帧开始
    - 使用随机stride前进
    - 到达边界后反向，并可选重新选择stride
    - 避免固定前进偏置

    用于论文4.2第二阶段长序列训练。
    """

    def __init__(
        self,
        total_frames: int,
        stride_range: Tuple[int, int] = (1, 3),
        redraw_stride_after_reverse: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            total_frames: 视频总帧数
            stride_range: stride范围 [min, max]
            redraw_stride_after_reverse: 反向后是否重新选择stride
            seed: 随机种子
        """
        self.total_frames = total_frames
        self.stride_range = stride_range
        self.redraw_stride_after_reverse = redraw_stride_after_reverse
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        print(f"[FoldbackVideoSampler] 初始化")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - stride范围: {stride_range}")
        print(f"  - 反向后重选stride: {redraw_stride_after_reverse}")

    def sample_sequence(
        self,
        num_views: int,
        start_frame: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> List[int]:
        """
        采样一个长序列

        Args:
            num_views: 需要采样的帧数
            start_frame: 起始帧（None则随机）
            stride: 前进stride（None则随机）

        Returns:
            frame_ids: 采样的帧ID列表
        """
        # 随机起始帧
        if start_frame is None:
            start_frame = random.randint(0, self.total_frames - 1)

        # 随机stride
        if stride is None:
            stride = random.randint(self.stride_range[0], self.stride_range[1])

        frame_ids = [start_frame]
        current_frame = start_frame
        direction = 1  # 1=前进, -1=后退

        while len(frame_ids) < num_views:
            next_frame = current_frame + direction * stride

            # 检查边界
            if next_frame < 0:
                # 到达左边界，反向
                direction = 1
                next_frame = 0
                if self.redraw_stride_after_reverse:
                    stride = random.randint(self.stride_range[0], self.stride_range[1])
            elif next_frame >= self.total_frames:
                # 到达右边界，反向
                direction = -1
                next_frame = self.total_frames - 1
                if self.redraw_stride_after_reverse:
                    stride = random.randint(self.stride_range[0], self.stride_range[1])

            # 避免重复帧
            if next_frame != current_frame:
                frame_ids.append(next_frame)
                current_frame = next_frame

        return frame_ids

    def sample_batch(
        self,
        batch_size: int,
        num_views_per_sample: List[int],
    ) -> List[List[int]]:
        """
        采样一批长序列

        Args:
            batch_size: batch大小
            num_views_per_sample: 每个样本的views数列表

        Returns:
            batch_frame_ids: 每个样本的frame_ids列表
        """
        batch_frame_ids = []

        for i in range(batch_size):
            num_views = num_views_per_sample[i]
            frame_ids = self.sample_sequence(num_views)
            batch_frame_ids.append(frame_ids)

        return batch_frame_ids


class ProgressiveViewCurriculum:
    """
    Progressive View Curriculum

    论文4.2的views数线性增长策略：
    - 开始时使用较少views（如24）
    - 随iteration增加到目标views（如320）
    - 线性增长schedule

    显存不足时可降级：
    - start: 8, end: 64
    - start: 16, end: 128
    """

    def __init__(
        self,
        views_start: int = 24,
        views_end: int = 320,
        total_iterations: int = 160000,
        warmup_iterations: int = 8000,  # 前5%不增长
        seed: int = 42,
    ):
        """
        Args:
            views_start: 起始views数
            views_end: 目标views数
            total_iterations: 总训练iterations
            warmup_iterations: warmup期间保持views_start
            seed: 随机种子
        """
        self.views_start = views_start
        self.views_end = views_end
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self.seed = seed

        random.seed(seed)

        print(f"[ProgressiveViewCurriculum] 初始化")
        print(f"  - Views: {views_start} -> {views_end}")
        print(f"  - 总iterations: {total_iterations}")
        print(f"  - Warmup iterations: {warmup_iterations}")

    def get_num_views(self, iteration: int) -> int:
        """
        根据iteration获取当前views数

        Args:
            iteration: 当前iteration

        Returns:
            num_views: 当前views数
        """
        # Warmup期间保持起始views
        if iteration < self.warmup_iterations:
            return self.views_start

        # 线性增长
        progress = (iteration - self.warmup_iterations) / (self.total_iterations - self.warmup_iterations)
        progress = min(1.0, progress)

        num_views = int(self.views_start + progress * (self.views_end - self.views_start))

        return num_views

    def get_num_views_with_variance(self, iteration: int, variance: int = 4) -> int:
        """
        获取views数并添加随机波动

        Args:
            iteration: 当前iteration
            variance: 随机波动范围

        Returns:
            num_views: 带波动的views数
        """
        base_views = self.get_num_views(iteration)
        delta = random.randint(-variance, variance)
        num_views = max(self.views_start, min(self.views_end, base_views + delta))

        return num_views


class LocalWindowSampler:
    """
    Local Window Sampler for Relative Pose Loss

    论文4.2的局部窗口采样：
    - GCA的局部参考窗口k在[16, 64]中随机采样
    - 用于relative pose loss只计算局部窗口内的pairs
    - 每个iteration随机选择窗口大小

    显存不足时可降级：
    - k_range: [8, 32]
    """

    def __init__(
        self,
        k_min: int = 16,
        k_max: int = 64,
        seed: int = 42,
    ):
        """
        Args:
            k_min: 最小窗口大小
            k_max: 最大窗口大小
            seed: 随机种子
        """
        self.k_min = k_min
        self.k_max = k_max
        self.seed = seed

        random.seed(seed)

        print(f"[LocalWindowSampler] 初始化")
        print(f"  - 窗口范围: [{k_min}, {k_max}]")

    def sample_window_size(self) -> int:
        """随机采样窗口大小"""
        return random.randint(self.k_min, self.k_max)

    def get_window_pairs(self, num_views: int, window_size: int) -> List[Tuple[int, int]]:
        """
        获取窗口内的所有pairs

        Args:
            num_views: 总views数
            window_size: 窗口大小k

        Returns:
            pairs: (i, j) pairs列表，窗口内的相邻帧
        """
        pairs = []

        # 每个窗口内的pairs
        # 使用滑动窗口方式：每个窗口k帧，计算窗口内的relative pose
        for start in range(0, num_views - 1, window_size):
            window_end = min(start + window_size, num_views)

            # 窗口内所有pairs
            for i in range(start, window_end):
                for j in range(start, window_end):
                    if i != j:
                        pairs.append((i, j))

        return pairs

    def get_adjacent_pairs(self, num_views: int, window_size: int) -> List[Tuple[int, int]]:
        """
        获取相邻帧pairs（简化版，减少计算量）

        Args:
            num_views: 总views数
            window_size: 窗口大小

        Returns:
            pairs: 相邻帧pairs列表
        """
        pairs = []

        # 相邻帧pairs（距离<=window_size）
        for i in range(num_views):
            for j in range(i + 1, min(i + window_size + 1, num_views)):
                pairs.append((i, j))
                pairs.append((j, i))  # 双向

        return pairs


def create_foldback_sampler(total_frames: int, config: dict) -> FoldbackVideoSampler:
    """创建FoldbackVideoSampler"""
    return FoldbackVideoSampler(
        total_frames=total_frames,
        stride_range=config.get('stride_range', (1, 3)),
        redraw_stride_after_reverse=config.get('redraw_stride_after_reverse', True),
        seed=config.get('seed', 42),
    )


def create_view_curriculum(config: dict) -> ProgressiveViewCurriculum:
    """创建ProgressiveViewCurriculum"""
    return ProgressiveViewCurriculum(
        views_start=config.get('views_start', 24),
        views_end=config.get('views_end', 320),
        total_iterations=config.get('total_iterations', 160000),
        warmup_iterations=config.get('warmup_iterations', 8000),
        seed=config.get('seed', 42),
    )


def create_local_window_sampler(config: dict) -> LocalWindowSampler:
    """创建LocalWindowSampler"""
    return LocalWindowSampler(
        k_min=config.get('k_min', 16),
        k_max=config.get('k_max', 64),
        seed=config.get('seed', 42),
    )


if __name__ == '__main__':
    # 测试Foldback sampler
    print("\n" + "="*60)
    print("测试 FoldbackVideoSampler")
    print("="*60)

    sampler = FoldbackVideoSampler(total_frames=2000, stride_range=(1, 5))

    # 测试采样
    for num_views in [24, 64, 128]:
        frame_ids = sampler.sample_sequence(num_views)
        print(f"  采样 {num_views} views: start={frame_ids[0]}, end={frame_ids[-1]}, stride变化={len(set(np.diff(frame_ids)))}")

    # 测试Curriculum
    print("\n" + "="*60)
    print("测试 ProgressiveViewCurriculum")
    print("="*60)

    curriculum = ProgressiveViewCurriculum(
        views_start=24,
        views_end=320,
        total_iterations=160000,
    )

    for iter in [0, 8000, 40000, 80000, 120000, 160000]:
        views = curriculum.get_num_views(iter)
        print(f"  Iter {iter}: views={views}")

    # 测试Local Window
    print("\n" + "="*60)
    print("测试 LocalWindowSampler")
    print("="*60)

    window_sampler = LocalWindowSampler(k_min=16, k_max=64)

    k = window_sampler.sample_window_size()
    pairs = window_sampler.get_adjacent_pairs(100, k)
    print(f"  窗口大小k={k}, views=100, pairs数={len(pairs)}")

    print("\n[测试完成]")