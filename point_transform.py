import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


def point_guided_deformation_mls(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Moving Least Squares (MLS) 图像变形
    提供更平滑和保角的变形效果
    """
    if len(source_pts) < 3:
        return np.array(image)

    h, w = image.shape[:2]

    # 创建网格（目标图像的像素坐标）
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)

    src = source_pts.astype(np.float32)
    dst = target_pts.astype(np.float32)

    warped_coords = np.zeros_like(coords)

    # 对每个目标像素，找到它在源图像中的对应位置
    for i, v in enumerate(coords):
        # v 是目标图像中的点
        # 我们需要找到它在源图像中的位置

        # 计算到目标控制点的权重
        diff = dst - v
        dist2 = np.sum(diff ** 2, axis=1)
        weights = 1.0 / (dist2 + eps) ** alpha
        total_weight = np.sum(weights)

        if total_weight < eps:
            warped_coords[i] = v
            continue

        # 加权中心（在目标空间中）
        q_star = np.sum(weights[:, np.newaxis] * dst, axis=0) / total_weight
        # 加权中心（在源空间中）
        p_star = np.sum(weights[:, np.newaxis] * src, axis=0) / total_weight

        # 中心化坐标
        q_hat = dst - q_star
        p_hat = src - p_star

        # 构建加权协方差矩阵
        A = np.zeros((2, 2))
        for j in range(len(src)):
            wj = weights[j]
            # 注意：这里是 p_hat 和 q_hat 的外积
            A += wj * np.outer(p_hat[j], q_hat[j])

        # SVD分解得到最优旋转
        U, S, Vt = np.linalg.svd(A)
        R = U @ Vt

        # 计算变换矩阵
        M = R

        # 计算源图像中的对应点
        # v 是目标点，我们想要找到它在源图像中的位置
        # 变换公式：p = (v - q_star) @ M + p_star
        warped_coords[i] = (v - q_star) @ M + p_star

    # 重塑坐标
    warped_coords = warped_coords.reshape(h, w, 2).astype(np.float32)

    # 使用remap进行插值
    map_x = warped_coords[:, :, 0]
    map_y = warped_coords[:, :, 1]

    # 确保坐标在有效范围内
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # 反向映射：从源图像采样到目标图像
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return warped.astype(np.uint8)


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Point-guided image deformation using MLS algorithm
    """
    # 确保有足够的控制点
    if len(source_pts) < 3:
        print(f"Warning: Need at least 3 control points, got {len(source_pts)}")
        return np.array(image)

    print(f"Applying MLS deformation with {len(source_pts)} control points")

    # 执行MLS变形
    warped_image = point_guided_deformation_mls(image, source_pts, target_pts, alpha, eps)

    return warped_image


def run_warping():
    """执行图像变形"""
    global points_src, points_dst, image

    if image is None:
        return None

    if len(points_src) < 3:
        print(f"Please select at least 3 control point pairs. Current: {len(points_src)}")
        return image

    if len(points_src) != len(points_dst):
        print(f"Mismatch: Source points {len(points_src)}, Target points {len(points_dst)}")
        return image

    print(f"Source points: {points_src}")
    print(f"Target points: {points_dst}")

    try:
        warped_image = point_guided_deformation(
            image,
            np.array(points_src, dtype=np.float32),
            np.array(points_dst, dtype=np.float32)
        )
        return warped_image
    except Exception as e:
        print(f"Error during warping: {e}")
        return image


# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()