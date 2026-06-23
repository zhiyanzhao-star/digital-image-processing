# Assignment 2 - DIP with PyTorch

本仓库为数字图像处理（DIP）课程作业 2 的实现，包含 **泊松图像编辑（Poisson Image Editing）** 和 **Pix2Pix 图像翻译** 两个任务，均基于 PyTorch 框架。

---

## 环境配置

推荐使用 conda 环境：

```bash
conda create -n dip python=3.10
conda activate dip
pip install torch torchvision opencv-python gradio pillow numpy
```

主要依赖：PyTorch >= 2.0, OpenCV >= 4.0, Gradio >= 3.0, PIL, NumPy

---

## Work 1: Poisson Image Blending（泊松图像编辑）

### 原理

泊松图像编辑（Pérez et al., 2003）基于泊松偏微分方程，在将源图像区域无缝混合到目标图像时，保持源区域的梯度信息。核心思想是最小化混合区域内的梯度差异：

$$\min_f \iint_{\Omega} |\nabla f - \nabla g|^2, \quad \text{s.t.} \quad f|_{\partial\Omega} = f^*|_{\partial\Omega}$$

其中 $g$ 为源图像（前景），$f^*$ 为目标图像（背景），$\Omega$ 为混合区域。

### 实现

在 [run_blending_gradio.py](run_blending_gradio.py) 中完成了两个关键函数：

#### 1. `create_mask_from_points` — 多边形掩码生成

使用 PIL 的 `ImageDraw.Draw.polygon` 将用户选择的多边形顶点转换为二值掩码：
- 多边形内部像素值为 **255**（前景区域）
- 多边形外部像素值为 **0**（背景区域）

```python
mask_img = Image.new('L', (img_w, img_h), 0)
draw = ImageDraw.Draw(mask_img)
draw.polygon(polygon_points, fill=255)
mask = np.array(mask_img)
```

#### 2. `cal_laplacian_loss` — 拉普拉斯损失计算

使用 `torch.nn.functional.conv2d` 和 3×3 拉普拉斯核对前景和混合图像分别计算拉普拉斯变换，然后在掩码区域内计算 L1 损失：

$$\mathcal{L} = \frac{1}{N} \sum |\Delta(fg) \odot M_{fg} - \Delta(blended) \odot M_{bg}|$$

拉普拉斯核通过 `groups=3` 的 group convolution 对 RGB 三通道分别计算。

#### 优化流程

1. 混合图像掩码区域初始化为前景与背景的加权平均
2. Adam 优化器（lr=0.01），5000 步迭代
3. 每步仅对掩码区域内像素反向传播梯度
4. 第 3333 步（2/3 处）学习率降至 0.001

### 使用方法

```bash
python run_blending_gradio.py
```

Gradio 界面操作流程：上传前景/背景图像 → 点击选择多边形 → Close Polygon → 调整偏移 → Blend Images

### 测试数据

`data_poisson/` 目录提供三组测试图像：

| 场景 | 前景 | 背景 |
|------|------|------|
| Equation | `equation/source.png` | `equation/target.png` |
| Mona Lisa | `monolisa/source.png` | `monolisa/target.png` |
| Water | `water/source.jpg` | `water/target.jpg` |

---

## Work 2: Pix2Pix Image-to-Image Translation

### 原理

Pix2Pix (Isola et al., 2017) 使用条件生成对抗网络进行图像到图像的翻译。本实现采用全卷积网络（FCN, Long et al., 2015）作为生成器，通过编码器-解码器架构学习从真实建筑照片到语义标签的映射。

### 网络架构：FullyConvNetwork

[FCN_network.py](Pix2Pix/FCN_network.py) 中实现的编码器-解码器结构（6 层编码 + 6 层解码）：

| 层 | 类型 | 通道变化 | 空间尺寸 | 激活 |
|----|------|---------|---------|------|
| conv1 | Conv2d (k=4, s=2, p=1) | 3 → 8 | 256 → 128 | BN + ReLU |
| conv2 | Conv2d (k=4, s=2, p=1) | 8 → 16 | 128 → 64 | BN + ReLU |
| conv3 | Conv2d (k=4, s=2, p=1) | 16 → 32 | 64 → 32 | BN + ReLU |
| conv4 | Conv2d (k=4, s=2, p=1) | 32 → 64 | 32 → 16 | BN + ReLU |
| conv5 | Conv2d (k=4, s=2, p=1) | 64 → 128 | 16 → 8 | BN + ReLU |
| conv6 | Conv2d (k=4, s=2, p=1) | 128 → 256 | 8 → 4 | BN + ReLU |
| deconv1 | ConvTranspose2d (k=4, s=2, p=1) | 256 → 128 | 4 → 8 | BN + ReLU |
| deconv2 | ConvTranspose2d (k=4, s=2, p=1) | 128 → 64 | 8 → 16 | BN + ReLU |
| deconv3 | ConvTranspose2d (k=4, s=2, p=1) | 64 → 32 | 16 → 32 | BN + ReLU |
| deconv4 | ConvTranspose2d (k=4, s=2, p=1) | 32 → 16 | 32 → 64 | BN + ReLU |
| deconv5 | ConvTranspose2d (k=4, s=2, p=1) | 16 → 8 | 64 → 128 | BN + ReLU |
| deconv6 | ConvTranspose2d (k=4, s=2, p=1) | 8 → 3 | 128 → 256 | **Tanh** |

- 总参数量：**1,399,763**
- 最后一层 Tanh 激活，输出归一化到 [-1, 1]

### 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | [Facades (CMP)](https://cmp.felk.cvut.cz/~tylecr1/facade/) |
| 训练集 / 验证集 | 400 / 100 张 |
| 输入 / 输出尺寸 | 256 × 256 RGB |
| 损失函数 | L1 Loss |
| 优化器 | Adam (lr=1e-3, β₁=0.5, β₂=0.999) |
| 学习率调度 | StepLR (step=50, γ=0.2) |
| Batch Size | 16 |
| Epochs | 100 |
| 设备 | CPU (Intel) |

### 训练方法

```bash
cd Pix2Pix
bash download_facades_dataset.sh   # 下载数据集 (~30MB)
python train.py                     # 开始训练，约需 3-4 小时 (CPU)
```

### 训练结果

| 指标 | 初始值 (Epoch 1) | 最佳值 | 最终值 (Epoch 100) |
|------|------------------|--------|-------------------|
| Train Loss (L1) | 0.884 | 0.107 | 0.126 |
| Val Loss (L1) | 0.738 | 0.392 (Epoch 59) | 0.407 |

**Train Loss 变化趋势：**

| Epoch | 1 | 5 | 10 | 20 | 40 | 60 | 80 | 100 |
|-------|------|------|------|------|------|------|------|------|
| Loss | 0.714 | 0.383 | 0.337 | 0.306 | 0.234 | 0.188 | 0.165 | 0.126 |

**Val Loss 变化趋势：**

| Epoch | 1 | 5 | 10 | 20 | 30 | 59 (best) | 100 |
|-------|------|------|------|------|------|-----------|------|
| Loss | 0.596 | 0.420 | 0.406 | 0.404 | 0.392 | **0.392** | 0.407 |

### 结果分析

1. **训练损失**从 0.884 持续下降至 0.126（下降 86%），模型有效学习了输入到输出的映射
2. **验证损失**在约第 10 epoch 后趋于稳定（~0.40），后续在 0.39-0.41 之间小幅波动
3. 验证损失在第 59 epoch 达到最低值 0.392，之后略有上升，呈现轻微过拟合
4. 基础 FCN 架构（无 skip connection）泛化能力受限于模型容量与数据集规模

### 输出文件

| 文件/目录 | 内容 |
|-----------|------|
| `checkpoints/pix2pix_model_epoch_50.pth` | 第 50 epoch 模型权重 |
| `checkpoints/pix2pix_model_epoch_100.pth` | 最终模型权重 |
| `train_results/epoch_*/` | 20 组训练对比图 (input \| target \| output) |
| `val_results/epoch_*/` | 20 组验证对比图 (input \| target \| output) |

每张对比图为 256×768 的横向三连图（Input RGB | Target Semantic | Model Output）。

### 改进方向

- 采用 **U-Net 架构**（skip connection）替代纯 FCN，提升细节保留
- 引入 **PatchGAN 判别器** 实现完整 Pix2Pix 框架（cGAN + L1 loss）
- 使用其他数据集扩展训练（maps, cityscapes, edges2shoes 等）
- 在 GPU 上训练以增大 batch size 和 epoch 数

---

## 项目结构

```
02_DIPwithPyTorch/
├── README.md                        # 本报告
├── run_blending_gradio.py           # Work 1: 泊松图像编辑 (Gradio UI)
├── data_poisson/                    # Work 1: 测试数据
│   ├── equation/                    #   公式混合
│   ├── monolisa/                    #   蒙娜丽莎
│   └── water/                       #   水中物体
└── Pix2Pix/                         # Work 2: Pix2Pix 图像翻译
    ├── README.md
    ├── FCN_network.py               #   全卷积网络定义
    ├── facades_dataset.py           #   数据集加载（支持中文路径）
    ├── train.py                     #   训练脚本
    ├── download_facades_dataset.sh  #   数据集下载脚本
    ├── datasets/facades/            #   Facades 数据集 (train+val+test)
    ├── checkpoints/                 #   模型权重
    ├── train_results/               #   训练集结果可视化
    └── val_results/                 #   验证集结果可视化
```

---

## 参考资料

1. Pérez, P., Gangnet, M., & Blake, A. (2003). [Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf). *ACM Transactions on Graphics (SIGGRAPH)*.
2. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/). *CVPR 2017*.
3. Long, J., Shelhamer, E., & Darrell, T. (2015). [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038). *CVPR 2015*.
