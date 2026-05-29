# Assignment 3: Bundle Adjustment — 实验报告

> **作者**: [填写姓名] &nbsp; | &nbsp; **日期**: 2026-05-28

---

## 目录

- [Task 1: 基于 PyTorch 的 Bundle Adjustment 实现](#task1)
- [Task 2: 基于 COLMAP 的多视角 3D 重建](#task2)
- [结果对比与讨论](#comparison)

---

<a name="task1"></a>
## Task 1: 基于 PyTorch 的 Bundle Adjustment 实现

### 1.1 问题描述

从 50 个不同视角的 2D 观测点中，通过优化恢复出 **3D 点坐标、相机外参 (R, T) 和焦距 f**。共有 20,000 个 3D 点、50 个相机视角，观测总量为 805,089 个可见点。

### 1.2 方法

#### 1.2.1 投影模型

采用 pin-hole 相机模型，投影公式为：

$$u = -f \cdot \frac{X_c}{Z_c} + c_x, \quad v = f \cdot \frac{Y_c}{Z_c} + c_y$$

其中 $[X_c, Y_c, Z_c]^T = R \cdot [X, Y, Z]^T + T$，$c_x = c_y = 512$（图像尺寸 1024×1024）。

#### 1.2.2 参数化

| 参数 | 数量 | 初始化 |
|------|------|--------|
| 焦距 $f$ | 1 | $f = 900$（对应 FoV ≈ 60°） |
| 旋转（Euler 角 XYZ convention） | 50 × 3 = 150 | 全部初始化为 0（单位旋转矩阵） |
| 平移 $T$ | 50 × 3 = 150 | $[0, 0, -2.5]$（相机在物体正前方） |
| 3D 点坐标 | 20,000 × 3 = 60,000 | $\mathcal{N}(0, 0.1^2)$ 随机初始化 |

总计: **60,301 个可优化参数**。

#### 1.2.3 优化策略

- **优化器**: Adam，三组不同学习率
  - 3D 点: $\text{lr} = 0.05$
  - 相机参数 (Euler + T): $\text{lr} = 0.01$
  - 焦距: $\text{lr} = 0.005$
- **学习率调度**: MultiStepLR，在迭代 500、1000、1500 时衰减为原来的 0.5
- **损失函数**: Masked MSE (仅计算可见点的重投影误差)
- **迭代次数**: 2000
- **设备**: CPU (Intel)

#### 1.2.4 旋转矩阵参数化

自行实现 Euler 角到旋转矩阵的转换（XYZ convention），无需依赖 pytorch3d：

$$R = R_z(\gamma) \cdot R_y(\beta) \cdot R_x(\alpha)$$

### 1.3 结果

#### 1.3.1 优化收敛

| 指标 | 最终值 |
|------|--------|
| MSE Loss | **0.0121 px²** |
| MAE | **0.08 pixels** |
| RMSE | **0.11 pixels** |
| 优化后焦距 | **899.98** |
| 优化耗时 | **33.2 s** |

#### 1.3.2 Loss 曲线

![Loss Curve](task1/task1-output/loss_curve.png)

![MAE Curve](task1/task1-output/mae_curve.png)

- 前 400 次迭代: Loss 从 ~8000 快速下降到 < 1.0
- 400-600 次迭代: 进一步下降到 0.026
- 600 次迭代后: Loss 稳定在约 0.012，MAE 稳定在 ~0.08 px（亚像素精度）

#### 1.3.3 逐视角重投影误差

| 视角 | MAE (px) | Median (px) | Max (px) |
|------|----------|-------------|-----------|
| View 00 | 0.13 | 0.11 | 0.52 |
| View 12 | 0.07 | 0.05 | 0.50 |
| View 25 | 0.08 | 0.07 | 0.35 |
| View 37 | 0.07 | 0.04 | 0.49 |
| View 49 | 0.13 | 0.10 | 0.67 |

所有视角的 MAE 均在 0.13 px 以下，最大误差不超过 1 像素，表明优化收敛良好。

#### 1.3.4 3D 点云重建

![3D Point Cloud](task1/task1-output/point_cloud_views.png)

**3D 点云尺寸**:
- X: [-0.816, 0.713]（宽度 ~1.53 单位）
- Y: [-0.950, 1.043]（高度 ~1.99 单位）
- Z: [-0.389, 0.254]（深度 ~0.64 单位）

点云呈现清晰的人头形状，6 个视角均可辨认出面部轮廓和头部结构。

**输出文件**:
- `task1-output/reconstructed.obj` — 带 RGB 颜色的 3D 点云（可用 MeshLab 打开）
- `task1-output/cameras.npz` — 优化后的相机参数（焦距、Euler 角、平移向量）

### 1.4 代码结构

```
bundle_adjustment.py          # 主程序：数据加载 → 参数初始化 → 优化 → 结果保存
visualize_results.py          # 3D 点云多视角可视化
```

核心函数:
- `euler_to_rotation_matrix(euler)`: Euler 角 → 旋转矩阵 (XYZ)
- `project(pts_3d, R, T, f)`: 3D 点 → 2D 投影坐标

---

<a name="task2"></a>
## Task 2: 基于 COLMAP 的多视角 3D 重建

### 2.1 实验环境

- **COLMAP 版本**: 4.1.0.dev0 (Commit 5b76f53)
- **安装方式**: GitHub Release 预编译 Windows 二进制（`colmap-x64-windows-nocuda.zip`）
- **Python 接口**: pycolmap 3.12.5 (通过 conda-forge)
- **设备**: CPU（无 CUDA）

### 2.2 重建流程

按标准 COLMAP 流程执行：

| 步骤 | 命令/函数 | 耗时 |
|------|----------|------|
| 1. 特征提取 | `colmap feature_extractor` (SIFT, 单相机 PINHOLE) | ~2.5 s |
| 2. 特征匹配 | `colmap exhaustive_matcher` | ~1.7 s |
| 3. 稀疏重建 | `colmap mapper` (Incremental SfM + Global BA) | ~3.7 s |
| **合计** | | **~8 s** |

### 2.3 结果

#### 2.3.1 稀疏重建统计

| 指标 | 数值 |
|------|------|
| 注册图像 | **50 / 50**（100%） |
| 3D 点数量 | **1,610** |
| 估计焦距 | **fx = 887.27, fy = 870.28** |
| 主点 | cx = 512.0, cy = 512.0 |
| 相机模型 | PINHOLE |

#### 2.3.2 稀疏点云

![COLMAP Sparse Point Cloud](task2/task2-output/稀疏点云-截图.png)

#### 2.3.3 特征提取详情

每幅图像提取的 SIFT 特征点数量在 287-523 之间（中位数约 380），这些合成渲染图像由于表面光滑，纹理信息有限，因此特征点较少。尽管如此，COLMAP 仍成功注册了全部 50 张图像。

### 2.4 关于稠密重建

由于当前环境无 CUDA GPU，稠密重建（Image Undistortion → Patch Match Stereo → Stereo Fusion）在 CPU 上运行极慢（预计数小时），因此仅完成到稀疏重建步骤。稠密重建需在有 NVIDIA GPU 的机器上运行 `run_colmap.ps1` 的第 4-6 步。

### 2.5 代码与脚本

| 文件 | 说明 |
|------|------|
| `run_colmap.py` | Python 版本（推荐，使用 pycolmap） |
| `run_colmap.ps1` | Windows PowerShell 版本 |
| `run_colmap.sh` | Linux Bash 版本 |

---

<a name="comparison"></a>
## 结果对比与讨论

### Task 1 vs Task 2 焦距对比

| | Task 1 (PyTorch BA) | Task 2 (COLMAP) |
|---|---|---|
| **焦距** | **899.98** | **887.27 / 870.28** |
| 焦距模型 | 单一焦距 f (fx = fy) | 独立 fx, fy (非正方形像素) |
| 参数数量 | 60,301 | COLMAP 内部自动估计 |

两者焦距估计非常接近（相差约 1.4%），说明两种方法互相验证了结果的正确性。

### 3D 点数量差异分析

| | Task 1 | Task 2 |
|---|---|---|
| 3D 点数 | **20,000** | **1,610** |
| 原因 | 已知 2D 对应关系，所有点均已匹配 | 需从图像提取 SIFT 特征并匹配 |

Task 1 已知所有 2D-3D 对应关系，因此可以恢复全部 20,000 个点。Task 2 需从零开始检测特征并匹配，合成渲染图像纹理不足导致仅能提取约 1,600 个稀疏点。

### 重投影精度

- Task 1 的 MAE = **0.08 px**，达到亚像素精度，说明梯度下降优化非常精确
- COLMAP 内部的 Bundle Adjustment 同样将重投影误差优化到亚像素级别

### 总结

1. **Task 1 成功**: 用 PyTorch 从零实现了完整的 Bundle Adjustment，60,301 个参数在 2000 次迭代内收敛到亚像素精度
2. **Task 2 成功**: COLMAP 成功注册了全部 50 张图像，估计的焦距与 Task 1 结果高度一致
3. 两种方法互相验证，说明重建结果可靠
