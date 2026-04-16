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
