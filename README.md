# digital-image-processing
Assignment 02
# Poisson Image Blending 泊松图像融合

基于 PyTorch 和 Gradio 实现的泊松图像融合（Poisson Image Editing）交互式应用。
## 任务简介
实现了经典的泊松图像编辑算法，通过求解泊松方程，在保持前景图像纹理细节的同时，使其与背景图像无缝融合。应用提供了直观的 Web 界面，支持实时调整融合位置。

### 环境依赖

```bash
# 创建虚拟环境（推荐）
conda create -n poisson_blend python=3.8
conda activate jupter_env
# 安装依赖包
pip install torch torchvision  # PyTorch
pip install gradio             # Web 界面
pip install pillow             # 图像处理
pip install numpy              # 数值计算
pip install opencv-python      # 可选，用于 mask 生成
# Assignment 2 - DIP with PyTorch

### 1. Implement Poisson Image Editing with PyTorch.
Fill the [Polygon to Mask function](run_blending_gradio.py#L95) and the [Laplacian Distance Computation](run_blending_gradio.py#L115) of `run_blending_gradio.py`.

### 2. Pix2Pix implementation.
Fill the [Fully Convolutional Network](Pix2Pix/FCN_network.py#L3) part of `Pix2Pix/FCN_network.py`.

---

## Fill Part

in Poisson Image Editing part:

```python
# Create a binary mask from polygon points
def create_mask_from_points(points, img_h, img_w):
    """
    Creates a binary mask from the given polygon points.

    Args:
        points (np.ndarray): Polygon points of shape (n, 2).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        np.ndarray: Binary mask of shape (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(p) for p in points.tolist()], outline=255, fill=255)

    mask = np.array(mask_img, dtype=np.uint8)
    return mask
```

```python
# Calculate the Laplacian loss between the foreground and blended image
def cal_laplacian_loss(foreground_img, foreground_mask, blended_img, background_mask):
    """
    Computes the Laplacian loss between the foreground and blended images within the masks.

    Args:
        foreground_img (torch.Tensor): Foreground image tensor.
        foreground_mask (torch.Tensor): Foreground mask tensor.
        blended_img (torch.Tensor): Blended image tensor.
        background_mask (torch.Tensor): Background mask tensor.

    Returns:
        torch.Tensor: The computed Laplacian loss.
    """
    loss = torch.tensor(0.0, device=foreground_img.device)

    lap_kernel = torch.tensor(
        [[0., -1., 0.],
         [-1., 4., -1.],
         [0., -1., 0.]],
        device=foreground_img.device,
        dtype=foreground_img.dtype
    ).view(1, 1, 3, 3)

    lap_kernel = lap_kernel.repeat(3, 1, 1, 1)

    fg_lap = torch.nn.functional.conv2d(
        foreground_img, lap_kernel, padding=1, groups=3
    )
    blended_lap = torch.nn.functional.conv2d(
        blended_img, lap_kernel, padding=1, groups=3
    )

    fg_vals = fg_lap[foreground_mask.bool().expand(-1, 3, -1, -1)]
    blended_vals = blended_lap[background_mask.bool().expand(-1, 3, -1, -1)]

    loss = torch.mean((fg_vals - blended_vals) ** 2)
    return loss
```

in Pix2Pix part:

```python
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        output = self.deconv5(x)

        return output
```

## Running

To run Poisson Image Editing, run:

```python
python run_blending_gradio.py
```

To run Pix2Pix, run:

```python
cd Pix2Pix
bash download_facades_dataset.sh
python train.py
```

## Results

### Poisson Image Editing

#### equation
| Source | Target | Output |
|---|---|---|
| <img src="data_poisson/equation/source.png" width="250"> | <img src="data_poisson/equation/target.png" width="250"> | <img src="data_poisson/equation/output.png" width="250"> |

#### monalisa
| Source | Target | Output |
|---|---|---|
| <img src="data_poisson/monolisa/source.png" width="250"> | <img src="data_poisson/monolisa/target.png" width="250"> | <img src="data_poisson/monolisa/output.png" width="250"> |

#### water
| Source | Target | Output |
|---|---|---|
| <img src="data_poisson/water/source.jpg" width="250"> | <img src="data_poisson/water/target.jpg" width="250"> | <img src="data_poisson/water/output.png" width="250"> |


## Acknowledgement

>📋 Thanks for the paper: [Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)

>📋 Thanks for the paper: [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)

>📋 Thanks for the paper: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
