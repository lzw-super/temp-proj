# LZW-Map Pose Metrics Guide

本文说明 `validate_lzw_map_stage1.py` 和 `validate_lzw_map_stage2.py` 中用于评估位姿的指标，重点解释：

```text
try_train/checkpoints/lzw_map_stage2_full_15-20_k4-8_5000iters/vis_validation_stage2/pose_metric_comparison.png
```

中的每个子图。所有验证结果默认比较三类模型：

- `LingBot Original`：原始预训练 LingBot-Map。
- `LZW Stage2 Init` / `LZW Init`：LZW 结构的初始化模型。
- `LZW Stage2 Trained` / `LZW Trained`：训练后的 LZW 模型。

## 总体判断原则

| 指标 | 越大越好 | 越小越好 | 单位 | 主要用途 |
| --- | --- | --- | --- | --- |
| `AUC@3/5/15/30` | yes | no | `%` | 相对位姿整体准确率，最适合做模型间主对比 |
| `Racc@k` | yes | no | `%` | 旋转误差低于 `k` 度的 pair 比例 |
| `Tacc@k` | yes | no | `%` | 平移方向误差低于 `k` 度的 pair 比例 |
| `ATE` | no | yes | `m` | 全局轨迹一致性、长程漂移 |
| `RPE-rot` | no | yes | `deg` | 相对旋转的局部一致性 |
| `RPE-trans` | no | yes | `m` | 相对平移的局部一致性 |
| `Anchored Rot` | no | yes | `deg` | 去掉第一帧姿态偏置后的旋转一致性 |
| `Legacy WXYZ` 指标 | no | yes | `deg` / loss | 诊断旧 quaternion 解释方式，不作为官方主指标 |

主结论建议优先看：

1. `Official XYZW AUC@3` 和 `AUC@30`：越高越好。
2. `Sim(3) ATE RMSE`：越低说明全局轨迹越稳。
3. `Official XYZW RPE Rot / RPE Trans`：越低说明局部相对运动越准。
4. `Legacy WXYZ` 只用于排查训练损失或 quaternion 顺序是否错配。

## Quaternion Convention

当前验证同时报告两套 quaternion 解释：

- `Official XYZW`：LingBot 官方 pose encoding，四元数顺序是 `x, y, z, w`，也就是 scalar-last。正式比较时应以这一套为准。
- `Legacy WXYZ`：旧本地训练损失曾把同一组四元数槽位解释为 `w, x, y, z`，也就是 scalar-first。它保留为诊断项，用来判断训练是否沿用了旧损失的错误约定。

如果 `Legacy WXYZ` 明显优于 `Official XYZW`，通常说明模型可能学到了旧损失定义下的旋转表示，而不是官方 LingBot pose encoding。这种情况不能简单认为位姿真实变好，需要进一步检查训练时的 quaternion 顺序和 loss 实现。

## AUC@3 / AUC@30

### 含义

`AUC@k` 是 pairwise relative-pose accuracy curve 在 `k` 度阈值内的面积，单位为 `%`，越高越好。

验证时会对一个 sample 内的所有无序帧对 `(i, j)` 计算相对位姿误差：

```text
relative_pose_pred = inv(T_pred_i) @ T_pred_j
relative_pose_gt   = inv(T_gt_i)   @ T_gt_j
```

然后分别计算：

- `rotation_error_deg`：预测相对旋转与 GT 相对旋转之间的角度误差。
- `translation_error_deg`：预测相对平移方向与 GT 相对平移方向之间的角度误差。这里比较方向，所以对尺度不敏感。

合成单个 pair 的 pose error：

```text
pose_error_deg = max(rotation_error_deg, translation_error_deg)
```

再统计从 `0` 到 `k` 度阈值下的累计召回曲线面积。代码里输出 `AUC@3/5/15/30`：

- `AUC@3`：严格指标。只有旋转和平移方向都非常准时才高，适合看精细位姿。
- `AUC@30`：宽松指标。能反映模型是否大体预测对了相对运动，适合看整体可用性。

### 当前图片中的情况

当前已有的 `pose_metric_comparison.png` 是旧版验证图，生成时间早于 AUC 指标接入，所以图片里还没有 `AUC@3` 和 `AUC@30` 子图。

更新后的验证脚本重新运行后，新的 `pose_metric_comparison.png` 会扩展为 2x4 图，新增：

- `Official XYZW AUC@3 (%)`
- `Official XYZW AUC@30 (%)`

终端表格和 JSON 里也会包含 `AUC@5`、`AUC@15`、`Racc@k`、`Tacc@k`。

## Racc@k / Tacc@k

这两个指标不在旧版图片中，但会写入新验证结果。

- `Racc@k`：所有帧对中，`rotation_error_deg < k` 的比例。
- `Tacc@k`：所有帧对中，`translation_error_deg < k` 的比例。

它们是 AUC 的拆解项：

- 如果 `Racc` 高但 `Tacc` 低，说明旋转还行，平移方向差。
- 如果 `Tacc` 高但 `Racc` 低，说明平移方向还行，旋转差。
- 如果二者都高，`AUC@k` 通常也会高。

## Sim(3) ATE RMSE (m)

图片子图名：

```text
Sim(3) ATE RMSE (m)
```

### 含义

`ATE` 是 Absolute Trajectory Error，用于评估整条预测轨迹与 GT 轨迹的全局一致性。

由于单目/多视图网络可能存在尺度、坐标系和整体旋转差异，验证时先对预测相机中心做 Umeyama `Sim(3)` 对齐：

```text
aligned_center_pred = scale * R * center_pred + t
```

然后计算每帧相机中心和 GT 相机中心之间的欧氏距离，并取 RMSE：

```text
ATE_RMSE = sqrt(mean(||aligned_center_pred - center_gt||^2))
```

单位是米，越小越好。

### 怎么读

这个指标主要看全局轨迹是否漂移。如果 ATE 很低，说明整条轨迹在 Sim(3) 对齐后整体贴近 GT；如果 ATE 高，说明全局轨迹形状、尺度或长程一致性较差。

当前 Stage2 图中：

| 模型 | ATE RMSE |
| --- | ---: |
| LingBot Original | `0.0063 m` |
| LZW Stage2 Init | `0.1056 m` |
| LZW Stage2 Trained | `0.0977 m` |

这说明训练后的 Stage2 相比初始化 ATE 略有改善，但仍明显弱于原始 LingBot。

## Official XYZW Anchored Rot (deg)

图片子图名：

```text
Official XYZW Anchored Rot (deg)
```

### 含义

该指标用官方 `XYZW` quaternion 解释预测旋转，并把第一帧作为旋转参考，评估每个后续 view 相对第一帧的旋转误差。

代码逻辑近似为：

```text
anchor_rot_pred_i = R_pred_0^T @ R_pred_i
anchor_rot_gt_i   = R_gt_i
error_i = angle(anchor_rot_pred_i, anchor_rot_gt_i)
```

因为 ReplicaDataset 已经把 GT pose 归一化到第一帧坐标系，所以 `R_gt_0` 是 identity，后续 GT rotation 都可以看作相对第一帧的旋转。

单位是度，越小越好。

### 怎么读

这个指标剥离了一部分“第一帧整体旋转偏置”的影响，更关注后续帧相对 reference frame 的旋转是否一致。

当前 Stage2 图中：

| 模型 | Official XYZW Anchored Rot |
| --- | ---: |
| LingBot Original | `0.1611 deg` |
| LZW Stage2 Init | `61.7775 deg` |
| LZW Stage2 Trained | `10.1126 deg` |

这说明 Stage2 训练显著改善了官方 XYZW 旋转，但仍与原始 LingBot 有较大差距。

## Official XYZW RPE Rot (deg)

图片子图名：

```text
Official XYZW RPE Rot (deg)
```

### 含义

`RPE Rot` 是 Relative Pose Error for rotation，用官方 `XYZW` quaternion 解释预测旋转，对所有帧对 `(i, j)` 计算相对旋转误差：

```text
rel_rot_pred = R_pred_i^T @ R_pred_j
rel_rot_gt   = R_gt_i^T   @ R_gt_j
error_ij = angle(rel_rot_pred, rel_rot_gt)
```

当前代码对这些 pair 的角度误差取 mean。单位是度，越小越好。

### 怎么读

它反映的是相对旋转的局部/成对一致性。相比绝对旋转，它对全局坐标系偏置没那么敏感，更适合判断帧间运动是否学对。

当前 Stage2 图中：

| 模型 | Official XYZW RPE Rot |
| --- | ---: |
| LingBot Original | `0.1690 deg` |
| LZW Stage2 Init | `61.0232 deg` |
| LZW Stage2 Trained | `9.4741 deg` |

这说明 Stage2 训练让相对旋转从初始化的大误差降到了约 `9.47 deg`，但仍显著弱于原始 LingBot。

## Official XYZW RPE Trans RMSE (m)

图片子图名：

```text
Official XYZW RPE Trans RMSE (m)
```

### 含义

`RPE Trans` 是 Relative Pose Error for translation。它对所有帧对 `(i, j)` 比较相对平移向量：

```text
rel_trans_pred = sim3_scale * R_pred_i^T @ (center_pred_j - center_pred_i)
rel_trans_gt   =              R_gt_i^T   @ (center_gt_j   - center_gt_i)
error_ij = ||rel_trans_pred - rel_trans_gt||
```

这里预测平移会乘以 Sim(3) 对齐得到的尺度 `sim3_scale`，目的是在比较相对平移时消除整体尺度差异。单位是米，越小越好。

### 怎么读

它反映帧间平移幅度和方向的综合误差。如果这个值低，说明模型不仅大体走向正确，相邻/成对视角之间的相对位移也比较准。

当前 Stage2 图中：

| 模型 | Official XYZW RPE Trans RMSE |
| --- | ---: |
| LingBot Original | `0.0103 m` |
| LZW Stage2 Init | `0.2292 m` |
| LZW Stage2 Trained | `0.2913 m` |

这里可以看到训练后的 Stage2 在旋转上明显变好，但相对平移 RMSE 反而比初始化更差。这通常说明训练主要修正了旋转表示或旋转头，但平移尺度/相对位移仍不稳定，需要结合 ATE、RPE-trans、AUC 的 translation component 继续判断。

## Legacy WXYZ Anchored Rot (deg)

图片子图名：

```text
Legacy WXYZ Anchored Rot (deg)
```

### 含义

这和 `Official XYZW Anchored Rot` 的计算方式相同，但把预测 quaternion 解释为 `WXYZ`。

它不是 LingBot 官方 pose encoding，只是旧本地训练损失的诊断口径。

### 怎么读

如果 `Legacy WXYZ` 指标显著低于 `Official XYZW`，说明模型输出可能更符合旧损失中的错误 quaternion 顺序。此时应优先检查训练代码，不建议直接把 `Legacy WXYZ` 当作正式位姿效果。

当前 Stage2 图中：

| 模型 | Legacy WXYZ Anchored Rot |
| --- | ---: |
| LingBot Original | `12.0113 deg` |
| LZW Stage2 Init | `56.6422 deg` |
| LZW Stage2 Trained | `4.8731 deg` |

训练后的 `Legacy WXYZ` 比 `Official XYZW` 更低，这提示 Stage2 训练可能仍受旧 quaternion convention 影响。

## Legacy WXYZ RPE Rot (deg)

图片子图名：

```text
Legacy WXYZ RPE Rot (deg)
```

### 含义

这和 `Official XYZW RPE Rot` 的计算方式相同，但把预测 quaternion 解释为 `WXYZ`。

单位是度，越小越好，但只作为诊断项。

当前 Stage2 图中：

| 模型 | Legacy WXYZ RPE Rot |
| --- | ---: |
| LingBot Original | `12.0388 deg` |
| LZW Stage2 Init | `55.9802 deg` |
| LZW Stage2 Trained | `3.6008 deg` |

这再次显示训练后模型在旧 `WXYZ` 解释下旋转误差更低，说明后续训练/验证时应重点统一 quaternion convention。

## 其他 JSON / 终端位姿指标

除 `pose_metric_comparison.png` 中画出的指标外，验证脚本还会保存更多位姿指标到：

```text
pose_metrics_per_sample.json
lzw_map_validation_results.json
lzw_map_stage2_validation_results.json
```

常见字段如下。

### `pose_ate_sim3_mean_m` / `pose_ate_sim3_max_m`

- `mean`：Sim(3) 对齐后逐帧中心误差的平均值。
- `max`：Sim(3) 对齐后逐帧中心误差最大值。
- 都是米，越小越好。

`RMSE` 更容易受大误差影响，`mean` 更平滑，`max` 用来发现极端坏帧。

### `pose_sim3_scale`

Sim(3) 对齐时估计出的整体尺度。用于了解预测轨迹尺度和 GT 尺度之间的比例关系。

如果该值长期偏离合理范围，说明模型预测的相机中心尺度存在系统偏差。

### `pose_pred_center_norm_mean_m` / `pose_gt_center_norm_mean_m`

分别统计预测相机中心和 GT 相机中心相对第一帧的平均距离。

它们用于粗看预测轨迹尺度是否和 GT 接近。如果预测值远小于 GT，说明轨迹收缩；远大于 GT，说明轨迹放大。

### `pose_anchor_raw_scale`

用第一帧作为 anchor，只拟合一个平移尺度时得到的原始 scale：

```text
scale = sum(pred_center * gt_center) / sum(pred_center^2)
```

可能为负。负值通常表示预测平移方向整体反了。

### `pose_anchor_positive_scale`

把 `pose_anchor_raw_scale` clamp 到正数后的尺度，用于旧的 scale-aligned pose loss 和诊断。

### `pose_anchor_negative_scale_fraction`

`raw_scale < 0` 的 batch 比例。越高说明越多样本存在整体平移方向反向的问题。

### `pose_anchor_trans_rmse_m` / `pose_anchor_trans_mean_m`

第一帧 anchor + 一个正平移尺度对齐后，相机中心平移误差的 RMSE / mean。单位米，越小越好。

它比 Sim(3) ATE 更严格，因为它不允许任意 3D 旋转对齐，只允许以第一帧为 anchor 做尺度校正。

### `pose_xyzw_abs_rot_mean_deg`

不做第一帧旋转对齐，直接比较预测旋转和 GT 旋转的平均角度误差。单位度，越小越好。

这个指标对全局旋转偏置敏感。如果 `abs_rot` 高但 `anchor_rot` 低，说明整体 reference frame 偏了，但相对旋转可能还可以。

### `pose_xyzw_rpe_trans_dir_mean_deg`

相对平移方向的平均角度误差，只看方向，不看长度。单位度，越小越好。

当 `RPE Trans RMSE` 高时，可以用它判断问题来自方向错误还是距离/尺度错误：

- 方向误差也高：平移方向预测错。
- 方向误差低但 RMSE 高：方向大致正确，但尺度或幅度错。

### `pose_xyzw_rpe_center_dist_rmse_m`

预测帧对距离和 GT 帧对距离之间的 RMSE：

```text
error_ij = sim3_scale * ||center_pred_j - center_pred_i|| - ||center_gt_j - center_gt_i||
```

它主要看相机中心之间的距离结构是否正确，不依赖旋转坐标系。

### `scale_aligned_abs_pose_loss` / `scale_aligned_rel_pose_loss`

旧验证中使用的 loss 风格指标。它们适合观察训练目标是否下降，但不如 AUC / ATE / RPE 直观，也容易受到 quaternion convention 影响。模型之间正式比较时不建议只看 loss。

## 如何读当前 Stage2 图片

当前旧版 `pose_metric_comparison.png` 的主要现象是：

1. `LingBot Original` 在所有官方 XYZW 指标上最强：ATE、anchored rotation、RPE rotation、RPE translation 都很低。
2. `LZW Stage2 Init` 位姿基本不可用：官方旋转误差在 `61 deg` 左右，ATE 也明显较高。
3. `LZW Stage2 Trained` 相比初始化明显改善了旋转：`Official XYZW RPE Rot` 从 `61.02 deg` 降到 `9.47 deg`。
4. 训练后的平移相对误差没有同步改善：`Official XYZW RPE Trans RMSE` 从 `0.2292 m` 升到 `0.2913 m`。
5. `Legacy WXYZ` 下训练模型旋转误差更低，说明需要重点检查训练和验证的 quaternion 顺序是否完全统一。

因此，当前这张图更像是在说明：

- Stage2 训练确实学到了一部分旋转信息；
- 但它还没有达到原始 LingBot-Map 的官方位姿质量；
- 平移和 quaternion convention 是下一步优先排查的问题。

重新运行更新后的验证脚本后，建议优先用新图中的 `AUC@3/AUC@30` 来做最终可读性更强的比较。

