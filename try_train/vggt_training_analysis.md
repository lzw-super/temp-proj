# VGGT 训练框架分析文档

> 基于 [facebookresearch/vggt](https://github.com/facebookresearch/vggt) 代码库分析
> VGGT: Visual Geometry Grounded Transformer — CVPR 2025 最佳论文

---

## 目录

1. [项目概览](#1-项目概览)
2. [代码结构](#2-代码结构)
3. [模型架构](#3-模型架构)
4. [损失函数详解](#4-损失函数详解)
5. [优化器与学习率调度](#5-优化器与学习率调度)
6. [数据处理与增强](#6-数据处理与增强)
7. [训练流程](#7-训练流程)
8. [分布式训练](#8-分布式训练)
9. [检查点与恢复](#9-检查点与恢复)
10. [完整配置参数](#10-完整配置参数)
11. [与 LingBot-Map 的对比](#11-与-lingbot-map-的对比)

---

## 1. 项目概览

VGGT 是一个前馈式 3D 基础模型，可从图像序列中进行流式 3D 重建。它使用 Vision Transformer 架构，结合时序因果注意力和 KV 缓存实现高效的在线推理（约 20 FPS）。

**核心训练特点：**
- 使用 Hydra 配置框架管理所有超参数
- 单阶段训练（默认配置下同时训练 Camera + Depth Head）
- 支持 DDP 多 GPU 分布式训练
- 动态 batch 大小（根据每 batch 图像数量自适应调整）
- AMP 混合精度训练（bfloat16）
- 梯度裁剪（按模块分组）
- 支持 Aggregator 冻结训练（Head-Only 训练）

---

## 2. 代码结构

```
vggt/training/
├── config/
│   ├── default.yaml           # 主配置文件
│   └── default_dataset.yaml   # 数据集默认配置
├── data/
│   ├── augmentation.py        # 数据增强
│   ├── base_dataset.py        # 基础数据集类
│   ├── composed_dataset.py    # 组合数据集
│   ├── datasets/
│   │   ├── co3d.py            # CO3D 数据集
│   │   └── vkitti.py          # Virtual KITTI 数据集
│   ├── dataset_util.py        # 数据处理工具
│   ├── dynamic_dataloader.py  # 动态 batch 数据加载器
│   ├── track_util.py          # 轨迹工具
│   └── worker_fn.py           # DataLoader worker 初始化
├── train_utils/
│   ├── checkpoint.py          # 检查点保存/加载
│   ├── distributed.py         # 分布式工具
│   ├── freeze.py              # 模块冻结
│   ├── general.py             # 通用工具（数值稳定性等）
│   ├── gradient_clip.py       # 梯度裁剪
│   ├── logging.py             # 日志系统
│   ├── normalization.py       # 相机归一化
│   ├── optimizer.py           # 优化器封装
│   └── tb_writer.py           # TensorBoard 记录
├── launch.py                  # Hydra 启动入口
├── trainer.py                 # 主训练器
└── loss.py                    # 损失函数
```

---

## 3. 模型架构

### 3.1 VGGT 整体结构

**文件：** `vggt/models/vggt.py`

```python
class VGGT(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True,
                 enable_depth=True, enable_track=True):
        self.aggregator = Aggregator(img_size, patch_size, embed_dim)
        self.camera_head = CameraHead(dim_in=2*embed_dim)     # if enable_camera
        self.depth_head = DPTHead(dim_in=2*embed_dim, output_dim=2, ...)  # if enable_depth
        self.point_head = DPTHead(dim_in=2*embed_dim, output_dim=4, ...)  # if enable_point
        self.track_head = TrackHead(dim_in=2*embed_dim, ...)              # if enable_track
```

**前向传播流程：**

1. **Aggregator** 处理图像序列，输出 `aggregated_tokens_list` 和 `patch_start_idx`
2. 各 Head 独立从 `aggregated_tokens_list` 中提取所需 token 进行预测
3. Camera Head 和 Point/Depth Head 的输出在 `torch.cuda.amp.autocast(enabled=False)` 下计算（float32 精度）

**默认训练配置（仅 Camera + Depth）：**
```yaml
model:
  enable_camera: True
  enable_depth: True
  enable_point: False
  enable_track: False
```

### 3.2 Aggregator（Transformer 核心）

**文件：** `vggt/models/aggregator.py`

| 参数 | 值 | 说明 |
|------|-----|------|
| `embed_dim` | 1024 | 嵌入维度 |
| `depth` | 24 | Transformer 层数 |
| `num_heads` | 16 | 注意力头数 |
| `mlp_ratio` | 4.0 | MLP 扩展比例 |
| `num_register_tokens` | 4 | 注册 token 数量 |
| `patch_embed` | `dinov2_vitl14_reg` | DINOv2 ViT-L/14 骨干网络 |
| `aa_order` | `["frame", "global"]` | 交替注意力顺序 |
| `aa_block_size` | 1 | 交替注意力块大小 |
| `enable_3d_rope` | True | 3D 旋转位置编码 |

**关键特性：**
- **交替注意力 (Alternating Attention):** 在 "frame"（帧内自注意力）和 "global"（全局交叉注意力）之间交替执行
- **特殊 Token：** 2 个 Camera Token + 4 个 Register Token + Scale Token
- **RoPE：** 频率=100，用于时序位置编码
- **梯度检查点：** 训练时启用以节省显存
- **图像归一化：** ImageNet 均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]

### 3.3 CameraHead（迭代位姿预测）

**文件：** `vggt/heads/camera_head.py`

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim_in` | 2048 | 输入维度 (2 x embed_dim) |
| `trunk_depth` | 4 | 迭代细化次数 |
| `num_heads` | 16 | 注意力头数 |
| `pose_encoding_type` | `absT_quaR_FoV` | 9 维姿态编码 |
| `init_values` | 0.01 | LayerScale 初始值 |

**姿态编码格式 `absT_quaR_FoV`（9 维）：**
- 维度 0-2：绝对平移向量 (absT)
- 维度 3-6：四元数旋转 (quaR)
- 维度 7-8：焦距和偏移 (FoV)

**迭代细化机制（4 次迭代）：**

```
第 1 次迭代: 使用可学习的 empty_pose_tokens -> AdaLN 调制 -> trunk -> 输出 delta -> pred = delta
第 2-4 次迭代: detach(pred) -> embed_pose -> AdaLN 调制 -> trunk -> 输出 delta -> pred = pred + delta
```

- 每次迭代的 `pred_pose_enc` 都会 **detach**（截断梯度），避免 BPTT
- 使用 **AdaLN**（自适应层归一化）将上一步预测注入当前步
- 最终输出 `pose_enc_list` 包含 4 个阶段的预测（用于多阶段损失加权）

### 3.4 DPTHead（深度/点云预测）

**文件：** `vggt/heads/dpt_head.py`

| 参数 | Depth Head | Point Head |
|------|-----------|-----------|
| `output_dim` | 2 (depth + confidence) | 4 (xyz + confidence) |
| `activation` | `exp` | `inv_log` |
| `conf_activation` | `expp1` | `expp1` |

**DPT 结构：** 基于 DPT (Dense Prediction Transformer) 的多尺度特征融合，从不同层级的 Aggregator token 中提取特征进行密集预测。

---

## 4. 损失函数详解

**文件：** `training/loss.py`

### 4.1 多任务损失总览

```
Total Loss = Camera Loss x weight_camera + Depth Loss x weight_depth + Point Loss x weight_point
```

**默认权重：**
| 任务 | 权重 | 是否启用（默认） |
|------|------|-----------------|
| Camera | 5.0 | 是 |
| Depth | 1.0 | 是 |
| Point | 1.0 | 否 |
| Track | - | 否 |

### 4.2 Camera Loss（相机位姿损失）

**核心逻辑：**

```python
# 1. 有效帧过滤：只计算有效点 > 100 的帧
valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

# 2. 多阶段时序加权（gamma = 0.6）
for stage_idx in range(n_stages):  # n_stages = 4 (4次迭代)
    stage_weight = 0.6 ** (4 - stage_idx - 1)
    # stage 0: 0.6^3 = 0.216, stage 1: 0.6^2 = 0.36, stage 2: 0.6^1 = 0.6, stage 3: 0.6^0 = 1.0
    # -> 最后一次迭代权重最高，早期迭代权重低
```

**各分量损失：**

```python
# 使用 L1 损失（论文用 smooth L1，代码发现 L1 更稳定）
loss_T  = |pred[..., :3]  - gt[..., :3]|     # 平移损失，权重 1.0
loss_R  = |pred[..., 3:7] - gt[..., 3:7]|    # 旋转损失，权重 1.0
loss_FL = |pred[..., 7:]  - gt[..., 7:]|     # 焦距损失，权重 0.5

# 平移损失额外裁剪到 [-100, 100] 防止不稳定
loss_T = loss_T.clamp(max=100).mean()

# 最终相机损失
camera_loss = (avg_loss_T x 1.0 + avg_loss_R x 1.0 + avg_loss_FL x 0.5) x 5.0
```

### 4.3 Depth Loss（深度估计损失）

由三个子损失组成：

#### 4.3.1 置信度加权损失 (`loss_conf_depth`)

```python
# 核心公式：gamma x ||pred - gt||^2 x conf - alpha x log(conf)
loss_reg = ||gt[mask] - pred[mask]||   # L2 距离
loss_conf = gamma x loss_reg x conf[mask] - alpha x torch.log(conf[mask])
```

- `gamma = 1.0`：置信度损失权重
- `alpha = 0.2`：置信度正则化权重
- 鼓励模型对容易的样本赋予高置信度，对困难的赋予低置信度
- `-alpha x log(conf)` 项防止置信度趋向 0

#### 4.3.2 回归损失 (`loss_reg_depth`)

```python
loss_reg = ||gt[mask] - pred[mask]||   # 简单 L2 距离
```

#### 4.3.3 梯度损失 (`loss_grad_depth`)

```python
# 多尺度梯度损失 (scales=4)
for scale in range(4):
    step = 2^scale  # 1, 2, 4, 8
    diff = pred - gt
    grad_x = |diff[:, ::step, 1::step] - diff[:, ::step, :-1:step]|
    grad_y = |diff[:, 1::step, ::step] - diff[:, :-1:step, ::step]|
    # 可选：应用置信度加权

# 梯度裁剪到 max=100
grad_x = grad_x.clamp(max=100)
grad_y = grad_y.clamp(max=100)
```

#### 4.3.4 异常值过滤

```python
# 分位数过滤 (valid_range=0.98)
# 保留损失值 < 98% 分位数的样本，去除极端异常值
quantile_thresh = quantile(loss_tensor, 0.98)
loss_tensor = loss_tensor[loss_tensor < quantile_thresh]
```

### 4.4 Point Loss（3D 点云损失）

结构与 Depth Loss 完全相同，但作用于 3D 坐标：
- `loss_conf_point`：置信度加权损失
- `loss_reg_point`：回归损失
- `loss_grad_point`：梯度损失（**可选用法向量损失 `normal_loss` 代替标准梯度损失**）

**法向量损失（Point Head 推荐）：**

```python
# 从 3D 点云计算表面法向量（4 个方向的叉积）
n1 = cross(up_dir, left_dir)
n2 = cross(left_dir, down_dir)
n3 = cross(down_dir, right_dir)
n4 = cross(right_dir, up_dir)

# 损失 = 1 - cos(theta)，避免 arccos 数值不稳定
loss = 1 - dot(pred_normals, gt_normals)
```

### 4.5 Track Loss（轨迹跟踪损失，未清理）

代码中标记为 "dirty code"，暂未正式启用：

```python
# 序列损失 (sequence_loss)
# - 多阶段迭代加权 (gamma=0.8)
# - 可见性感知 (vis_aware)
# - BCE 可见性损失
# - 3像素阈值置信度
```

### 4.6 数值稳定性保障

```python
def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    # 1. 检测 nan/inf -> 替换为 0
    # 2. 硬裁剪到 [-hard_max, hard_max]
```

**在所有损失计算后都调用此函数**，防止 nan/inf 传播导致训练崩溃。

---

## 5. 优化器与学习率调度

### 5.1 优化器配置

```yaml
optim:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 5e-5
    weight_decay: 0.05
```

- **优化器：** AdamW
- **基础学习率：** 5e-5
- **权重衰减：** 0.05（恒定不变）

### 5.2 学习率调度策略

**线性热身 + 余弦衰减：**

```
+------------------------------------------------------+
|  5% 训练                    95% 训练                  |
|  线性热身                   余弦衰减                   |
|  1e-8 -> 5e-5               5e-5 -> 1e-8              |
+------------------------------------------------------+
```

**具体实现：**

```python
# fvcore CompositeParamScheduler
schedulers:
  - LinearParamScheduler:   start=1e-8, end=5e-5     # 5% 阶段
  - CosineParamScheduler:   start=5e-5, end=1e-8     # 95% 阶段
lengths: [0.05, 0.95]
interval_scaling: ['rescaled', 'rescaled']
```

**调度器更新时机：** 每个 batch 后更新（而非每个 epoch）

```python
exact_epoch = self.epoch + float(data_iter) / limit_train_batches
self.where = float(exact_epoch) / self.max_epochs  # 0.0 -> 1.0
optim.step_schedulers(self.where)
```

### 5.3 权重衰减调度

```yaml
weight_decay:
  - scheduler:
      _target_: fvcore.common.param_scheduler.ConstantParamScheduler
      value: 0.05
```

权重衰减在整个训练过程中保持恒定 0.05。

### 5.4 梯度裁剪

**按模块分组裁剪：**

```yaml
gradient_clip:
  configs:
    - module_name: ["aggregator"]   # Aggregator 模块
      max_norm: 1.0
      norm_type: 2                  # L2 范数
    - module_name: ["depth"]        # Depth Head
      max_norm: 1.0
      norm_type: 2
    - module_name: ["camera"]       # Camera Head
      max_norm: 1.0
      norm_type: 2
```

### 5.5 模块冻结

```yaml
frozen_module_names:
    - "*aggregator*"   # 冻结 Aggregator（示例配置）
```

冻结 Aggregator 即 **Head-Only 训练**，仅训练 Camera Head 和 Depth Head。

### 5.6 AMP 混合精度

```yaml
amp:
  enabled: True
  amp_dtype: bfloat16
```

- 使用 bfloat16 混合精度训练
- Camera Head 和 Point/Depth Head 的前向传播在 `autocast(enabled=False)` 下执行（float32 精度）
- 使用 `GradScaler` 进行缩放

---

## 6. 数据处理与增强

### 6.1 数据集

**主要数据集：** CO3D (Common Objects in 3D)

```yaml
# CO3D 数据集配置
Co3dDataset:
  split: train / test
  min_num_images: 24            # 最少图像数
  len_train: 100000             # 训练集长度（采样数）
  len_test: 10000               # 测试集长度
```

**辅助数据集：** Virtual KITTI (vkitti)

### 6.2 动态 Batch 采样

**文件：** `training/data/dynamic_dataloader.py`

VGGT 采用**动态 batch 大小**策略：根据每 batch 随机采样的图像数量和长宽比自适应调整 batch 大小。

```python
class DynamicBatchSampler(Sampler):
    # 训练时图像数量范围: [2, 24]
    # 验证时图像数量范围: [2, 12]
    # 长宽比范围: [0.33, 1.0]
    # 最大每 GPU 图像数: 48

    # batch_size = max_img_per_gpu / random_image_num
    # 例: 48 / 6 = 8 (每个 batch 8 个序列，每个序列 6 张图)
```

### 6.3 数据增强

```yaml
augs:
  cojitter: True              # 相同颜色抖动（所有帧同步）
  cojitter_ratio: 0.3         # 30% 概率应用
  scales: [0.8, 1.2]         # 尺度增强范围
  aspects: [0.33, 1.0]       # 长宽比范围
  color_jitter:
    brightness: 0.5           # 亮度抖动
    contrast: 0.5             # 对比度抖动
    saturation: 0.5           # 饱和度抖动
    hue: 0.1                  # 色调抖动
    p: 0.9                    # 90% 概率应用
  gray_scale: True            # 5% 概率灰度化
  gau_blur: False             # 高斯模糊（关闭）
```

**验证时不做增强：**

```yaml
# 验证集增强配置
augs:
  cojitter: False
  scales: null
  aspects: [1.0, 1.0]         # 固定长宽比
  color_jitter: null
  gray_scale: False
```

### 6.4 图像预处理流程

```
1. 随机尺度增强 (scales=[0.8, 1.2])
2. 基于主点裁剪 (crop by principal point)
3. 纵横比处理（可选旋转 90 度，landscape_check）
4. 调整大小到目标分辨率
5. 深度转 3D 坐标 (depth_to_world_coords_points)
```

### 6.5 相机归一化

**文件：** `training/train_utils/normalization.py`

```python
def normalize_camera_extrinsics_and_points_batch(
    extrinsics, cam_points, world_points, depths, point_masks):

    # 1. 将所有相机变换到第一个相机坐标系
    #    使用闭式 SE(3) 逆变换
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics[:, 0])

    # 2. 将所有 3D 点变换到第一个相机坐标系
    R = extrinsics[:, 0, :3, :3]
    t = extrinsics[:, 0, :3, 3]
    new_world_points = world_points @ R.T + t

    # 3. 按平均距离归一化（scale_by_points=True）
    avg_scale = (dist_sum / valid_count).clamp(min=1e-6, max=1e6)
    new_world_points = new_world_points / avg_scale
```

**关键点：**
- 所有坐标系相对于第一个相机
- 按点的平均距离进行缩放归一化
- 确保不同场景的尺度一致性

### 6.6 Batch 重复增强（可选）

```python
def _apply_batch_repetition(self, batch):
    # 将 batch 与其水平翻转版本拼接
    for key in tensor_keys:
        batch[key] = torch.cat([batch[key], torch.flip(batch[key], dims=[1])], dim=0)
```

默认关闭 (`repeat_batch: False`)。

---

## 7. 训练流程

### 7.1 训练主循环

```
run_train():
  while epoch < max_epochs (20):
    1. set_seeds(seed + epoch * 100 + distributed_rank)
    2. dataloader = train_dataset.get_loader(epoch=epoch + rank)
    3. train_epoch(dataloader)
    4. save_checkpoint(epoch)
    5. 清理内存 (gc.collect, cuda.empty_cache)
    6. if epoch % val_epoch_freq == 0: run_val()
    7. epoch += 1
  run_val()  # 最终验证
```

### 7.2 单 Epoch 训练流程

```
train_epoch(train_loader):
  model.train()
  gradient_clipper.setup_clipping(model)

  for data_iter, batch in enumerate(train_loader):
    if data_iter > limit_train_batches (800): break

    # 1. 数据预处理（相机归一化）
    batch = _process_batch(batch)   # float32 精度

    # 2. 复制到 GPU
    batch = copy_data_to_device(batch, device)

    # 3. 梯度累积分块
    chunked_batches = chunk_batch(batch, accum_steps=2)

    # 4. 前向 + 反向
    _run_steps_on_batch_chunks(chunked_batches):
      optim.zero_grad()
      for i, chunk in enumerate(chunks):
        with ddp_context:  # 最后一步同步梯度
          with amp.autocast(dtype=bfloat16):
            loss_dict = _step(chunk, model)    # 前向 + 损失
          loss = loss_dict["objective"] / accum_steps
          scaler.scale(loss).backward()         # 反向

    # 5. 学习率调度
    where = (epoch + data_iter/limit_train_batches) / max_epochs
    optim.step_schedulers(where)

    # 6. 梯度裁剪
    scaler.unscale_(optim.optimizer)
    gradient_clipper(model)

    # 7. 优化器步进
    scaler.step(optim.optimizer)
    scaler.update()
```

### 7.3 _step 方法（前向 + 损失计算）

```python
def _step(self, batch, model, phase, loss_meters):
    # 1. 前向传播
    y_hat = model(images=batch["images"])

    # 2. 损失计算
    loss_dict = self.loss(y_hat, batch)

    # 3. 记录日志
    log_data = {**y_hat, **loss_dict, **batch}
    self._update_and_log_scalars(log_data, ...)
    self._log_tb_visuals(log_data, ...)

    return loss_dict
```

### 7.4 验证流程

```
val_epoch(val_loader):
  model.eval()
  with torch.no_grad():
    for data_iter, batch in enumerate(val_loader):
      if data_iter > limit_val_batches (400): break
      with amp.autocast(dtype=bfloat16):
        val_loss_dict = _step(batch, model)
```

---

## 8. 分布式训练

### 8.1 DDP 配置

```yaml
distributed:
  backend: nccl
  find_unused_parameters: False
  timeout_mins: 30
  gradient_as_bucket_view: True   # 减少显存使用
  bucket_cap_mb: 25
  broadcast_buffers: True
```

### 8.2 CUDA 配置

```yaml
cuda:
  cudnn_deterministic: False
  cudnn_benchmark: False
  allow_tf32: True               # 允许 TF32 加速
```

### 8.3 分布式数据采样

```python
class DynamicDistributedSampler(DistributedSampler):
    def update_parameters(self, aspect_ratio, image_num):
        # 每个 epoch 可以动态更新采样参数
```

### 8.4 Worker 初始化

```python
def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    worker_seed = (rank * num_workers + worker_id + seed +
                   world_size * 12345 + epoch * 67890)
```

### 8.5 同步屏障

```python
dist.barrier()  # 初始化完成后、训练开始前同步所有进程
```

---

## 9. 检查点与恢复

### 9.1 检查点保存内容

```python
checkpoint = {
    "prev_epoch": epoch,
    "steps": {"train": steps, "val": steps},
    "time_elapsed": elapsed_time,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),    # if AMP enabled
}
```

### 9.2 保存策略

```yaml
checkpoint:
  save_dir: logs/${exp_name}/ckpts
  save_freq: 5                      # 每 5 个 epoch 保存一次
  resume_checkpoint_path: /PATH/CKPT  # 恢复路径
  strict: False                     # 非严格加载
```

- 每个 epoch 保存 `checkpoint.pt`（覆盖）
- 每 `save_freq` 个 epoch 额外保存 `checkpoint_{epoch}.pt`
- 只在 rank 0 保存

### 9.3 鲁棒保存

```python
def robust_torch_save(checkpoint, path):
    # 1. 旧文件重命名为 .bak
    # 2. 保存新文件
    # 3. 删除 .bak
    # 防止保存过程中崩溃导致文件损坏
```

### 9.4 自动恢复

```python
# 优先使用指定路径
if resume_checkpoint_path is not None:
    load(resume_checkpoint_path)
# 否则自动搜索最新检查点
else:
    ckpt_path = get_resume_checkpoint(save_dir)
    if ckpt_path: load(ckpt_path)
```

---

## 10. 完整配置参数

### 10.1 默认训练配置

```yaml
# ===== 基本参数 =====
exp_name: exp001
img_size: 518
patch_size: 14
num_workers: 8
seed_value: 42
accum_steps: 2             # 梯度累积步数（实际未使用，设为1时不累积）
max_img_per_gpu: 48        # 每 GPU 最大图像数
max_epochs: 20

limit_train_batches: 800   # 每 epoch 最多 800 个 batch
limit_val_batches: 400     # 验证最多 400 个 batch
val_epoch_freq: 5          # 每 5 个 epoch 验证一次

# ===== 模型 =====
model:
  _target_: vggt.models.vggt.VGGT
  enable_camera: True
  enable_depth: True
  enable_point: False
  enable_track: False

# ===== 损失 =====
loss:
  _target_: loss.MultitaskLoss
  camera:
    weight: 5.0
    loss_type: "l1"          # L1 比 smooth L1/L2 更稳定
    weight_trans: 1.0
    weight_rot: 1.0
    weight_focal: 0.5
    gamma: 0.6               # 多阶段衰减
  depth:
    weight: 1.0
    gradient_loss_fn: "grad" # 标准梯度损失
    valid_range: 0.98        # 分位数过滤
    gamma: 1.0
    alpha: 0.2
  point: null                # 默认不启用
  track: null                # 默认不启用

# ===== 优化器 =====
optim:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 5e-5
    weight_decay: 0.05
  frozen_module_names:
    - "*aggregator*"         # 冻结 Aggregator（Head-Only 训练）
  amp:
    enabled: True
    amp_dtype: bfloat16
  gradient_clip:
    configs:
      - module_name: ["aggregator"]
        max_norm: 1.0
        norm_type: 2
      - module_name: ["depth"]
        max_norm: 1.0
        norm_type: 2
      - module_name: ["camera"]
        max_norm: 1.0
        norm_type: 2
  options:
    lr:
      - scheduler: CompositeParamScheduler
          # 5% 线性热身: 1e-8 -> 5e-5
          # 95% 余弦衰减: 5e-5 -> 1e-8
          lengths: [0.05, 0.95]
    weight_decay:
      - scheduler: ConstantParamScheduler
          value: 0.05

# ===== 日志 =====
logging:
  log_dir: logs
  log_visuals: False
  log_freq: 1
  scalar_keys_to_log:
    train:
      keys_to_log:
        - loss_objective
        - loss_camera
        - loss_T / loss_R / loss_FL
        - loss_conf_depth / loss_reg_depth / loss_grad_depth
    val:
      keys_to_log: [同上]

# ===== 检查点 =====
checkpoint:
  save_dir: logs/${exp_name}/ckpts
  save_freq: 5
  strict: False

# ===== 分布式 =====
distributed:
  backend: nccl
  find_unused_parameters: False
  gradient_as_bucket_view: True
  bucket_cap_mb: 25
  broadcast_buffers: True
```

---

## 11. 与 LingBot-Map 的对比

| 特性 | VGGT | LingBot-Map |
|------|------|-------------|
| **训练框架** | Hydra + 自定义 Trainer | 自定义训练脚本 |
| **配置管理** | YAML + Hydra 实例化 | Python 代码内配置 |
| **损失函数** | L1 Camera + 置信度加权 Depth + 梯度损失 | 类似但具体实现有差异 |
| **Camera 迭代** | 4 次，detach 避免梯度回传 | 类似的迭代细化 |
| **置信度损失** | `gamma * loss * conf - alpha * log(conf)` | 需确认是否一致 |
| **相机归一化** | 第一相机坐标系 + 距离缩放 | 类似方案 |
| **学习率** | 5e-5, AdamW, 线性热身+余弦衰减 | 需确认 |
| **梯度裁剪** | 按模块分组 (max_norm=1.0) | 需确认 |
| **数据增强** | cojitter, color_jitter, 灰度化等 | 需确认 |
| **动态 Batch** | 根据图像数量自适应 | 固定 batch |
| **Aggregator 冻结** | 配置支持 `frozen_module_names` | 支持 Head-Only |
| **异常值处理** | 分位数过滤 + nan/inf 检查 | 需确认 |
| **多阶段训练** | Camera Head 内部迭代 (4 次) | Stage1 + Stage2 |

---

## 关键代码文件索引

| 功能 | 文件路径 | 关键行号 |
|------|----------|----------|
| 主训练器 | `training/trainer.py` | 46-868 |
| 训练主循环 | `training/trainer.py` | 377-401 |
| 单 Epoch 训练 | `training/trainer.py` | 501-636 |
| 前向+损失 | `training/trainer.py` | 738-758 |
| 相机归一化 | `training/trainer.py` | 716-736 |
| 损失函数 | `training/loss.py` | 16-78 |
| Camera Loss | `training/loss.py` | 81-155 |
| Depth Loss | `training/loss.py` | 239-278 |
| Point Loss | `training/loss.py` | 199-236 |
| 回归损失 | `training/loss.py` | 281-367 |
| 梯度损失 | `training/loss.py` | 370-508 |
| 法向量损失 | `training/loss.py` | 398-453 |
| 异常值过滤 | `training/loss.py` | 567-603 |
| VGGT 模型 | `vggt/models/vggt.py` | 17-97 |
| CameraHead | `vggt/heads/camera_head.py` | 19-149 |
| Aggregator | `vggt/models/aggregator.py` | - |
| DPT Head | `vggt/heads/dpt_head.py` | - |
| 主配置 | `training/config/default.yaml` | 1-184 |
| 数据集配置 | `training/config/default_dataset.yaml` | 1-80 |
| 优化器 | `training/train_utils/optimizer.py` | - |
| 梯度裁剪 | `training/train_utils/gradient_clip.py` | - |
| 检查点 | `training/train_utils/checkpoint.py` | - |
| 相机归一化 | `training/train_utils/normalization.py` | - |
| 数值稳定性 | `training/train_utils/general.py` | 29-57 |
| 动态数据加载 | `training/data/dynamic_dataloader.py` | 17-244 |
| CO3D 数据集 | `training/data/datasets/co3d.py` | 67-280 |
