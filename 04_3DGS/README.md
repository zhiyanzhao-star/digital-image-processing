# 简化版 3D Gaussian Splatting

本文为作业 4 的实现报告——基于纯 PyTorch 的简化版 3D Gaussian Splatting（3DGS）管线，通过 3D 高斯的可微光栅化实现多视角图像到 3D 场景的重建。

## 环境配置

**运行环境：**
- Python 3.12
- PyTorch 2.4.1（CPU）
- pycolmap 3.12.5
- opencv-python、numpy、natsort、tqdm

```setup
pip install torch torchvision opencv-python numpy natsort tqdm pycolmap
```

> 说明：原始代码依赖 `pytorch3d`，该库在 Windows Python 3.12 上不可用。本实现将 `pytorch3d.ops.knn.knn_points` 替换为纯 PyTorch KNN，将 `pytorch3d.ops.sample_farthest_points` 替换为纯 PyTorch 最远点采样（FPS）。

## 数据

```
data/
├── chair/images/   # 100 张多视角渲染图像（800×800）
└── lego/images/    # 100 张多视角渲染图像（800×800）
```

本报告以 **chair** 场景为例（NeRF 合成数据集）。

---

## 任务一：基于 COLMAP 的运动恢复结构（SfM）

运行 SfM 恢复相机内外参与稀疏 3D 点：

```bash
python mvs_with_pycolmap.py --data_dir data/chair
```

将 3D 点重投影到各视角图像进行验证：

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

### 结果

| 指标 | 数值 |
|------|------|
| 已注册图像 | 100 / 100 |
| 稀疏 3D 点 | 14,019 |
| 相机模型 | PINHOLE |
| 焦距 | 960 px |

稀疏点云已导出至 `data/chair/sparse/points3D.ply`（可用 MeshLab 查看）。

---

## 任务二：简化版 3D Gaussian Splatting

### 代码实现

根据 3DGS 论文实现了以下四个核心模块：

| 模块 | 文件位置 | 公式 |
|------|----------|------|
| 3D 协方差矩阵 | `gaussian_model.py:114` | $\Sigma = R S S^T R^T$ |
| 投影雅可比矩阵 | `gaussian_renderer.py:50-57` | $J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$ |
| 相机空间协方差 | `gaussian_renderer.py:59-60` | $\Sigma_{cam} = R \Sigma_{world} R^T$ |
| 2D 高斯取值 | `gaussian_renderer.py:84-93` | $f(\mathbf{x}) = \frac{1}{2\pi\sqrt{|\Sigma|}}\exp(-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu))$ |
| α-blending 渲染 | `gaussian_renderer.py:135-137` | $w_i = \alpha_i \prod_{j<i}(1-\alpha_j)$ |

### 训练

```train
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints \
    --num_epochs 10 --save_every 5 --debug_every 5 --device cpu
```

**超参数：**

| 参数 | 数值 |
|------|------|
| 高斯数量 | 3,000（从 14,019 个点中 FPS 采样） |
| 训练轮数 | 10 |
| 图像分辨率 | 100×100（8 倍下采样） |
| 学习率 | xyz: 1.6e-5, color: 0.025, opacity: 0.05, scale: 0.005, rotation: 0.001 |
| 优化器 | Adam |
| 批次大小 | 1 |
| 设备 | CPU |

### 训练结果

10 轮训练的 L1 损失变化（像素值范围 [0,1]）：

| 轮次 | 损失 |
|------|------|
| 0 | 0.1366 |
| 1 | 0.1082 |
| 2 | 0.0912 |
| 3 | 0.0854 |
| 4 | 0.0830 |
| 5 | 0.0818 |
| 6 | 0.0812 |
| 7 | 0.0808 |
| 8 | 0.0804 |
| 9 | 0.0800 |

损失在 10 轮内下降了 **41%**（0.137 → 0.080）。CPU 上每 10 轮训练耗时约 12 分钟。

### 训练过程可视化

调试图像保存位置：
- `data/chair/checkpoints/debug_images/epoch_0000.png` — 训练前
- `data/chair/checkpoints/debug_images/epoch_0005.png` — 5 轮后

### 渲染视频

```bash
python render_3dgs_mv.py --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000005.pt
```

输出路径：`data/chair/checkpoints/debug_rendering.mp4`

---

## 预训练模型

| 模型 | 轮次 | 损失 | 路径 |
|------|------|------|------|
| 初始模型 | 0 | 0.1366 | `data/chair/checkpoints/checkpoint_000000.pt` |
| 最优模型 | 5 | 0.0818 | `data/chair/checkpoints/checkpoint_000005.pt` |

---

## 任务三：与官方 3DGS 实现对比

本简化 PyTorch 实现与[官方 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)的主要差异：

| 方面 | 本实现 | 官方 3DGS |
|------|--------|-----------|
| **光栅化器** | 纯 PyTorch 逐像素计算 | CUDA 分块光栅化器 |
| **稠密化** | 未实现 | 自适应高斯稠密化 |
| **渲染速度** | ~0.7 s/图（CPU，100×100） | 实时（GPU，全分辨率） |
| **显存占用** | 所有高斯 × 所有像素同时存储 | 分块：仅加载可见高斯 |
| **渲染质量** | 有限（无稠密化、低分辨率） | 高（完整管线、高分辨率） |

### 差异分析

1. **训练速度**：官方 CUDA 光栅化器以 16×16 像素块为单位处理高斯，实现实时渲染。本实现中每个高斯对每个像素逐一求值，复杂度为 $O(N_{gaussians} \times H \times W)$，在无 GPU 情况下高分辨率场景不可行。

2. **渲染质量**：缺少自适应稠密化机制，高斯在训练过程中无法分裂或裁剪。3,000 个 FPS 采样点是固定的数量上限，点密度不足的区域会持续模糊。

3. **显存占用**：官方分块光栅化器仅将每个 16×16 块内的可见高斯加载到共享内存。本实现同时存储所有高斯与像素的交互量，显存消耗远高于分块方案。

4. **缺失模块**：本简化版未实现球谐函数（SH）视角相关颜色编码和自适应密度控制策略，这两个模块对恢复精细几何细节至关重要。
