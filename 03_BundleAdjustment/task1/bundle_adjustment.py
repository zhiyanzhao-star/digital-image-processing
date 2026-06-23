"""
Bundle Adjustment from scratch using PyTorch.
Recovers 3D points, camera extrinsics (R, T), and focal length from 2D observations.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# ==================== Configuration ====================
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

IMAGE_SIZE = 1024
CX = CY = IMAGE_SIZE / 2.0

N_ITERATIONS = 2000
LR_POINTS = 0.05
LR_CAMERAS = 0.01
LR_FOCAL = 0.005

# ==================== Load Data ====================
print("Loading data...")
data = np.load(f"{DATA_DIR}/points2d.npz")
view_keys = sorted(data.keys())
n_views = len(view_keys)
n_points = data[view_keys[0]].shape[0]

observations = np.stack([data[k] for k in view_keys], axis=0)
obs_tensor = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
visibility = obs_tensor[:, :, 2]
observed_2d = obs_tensor[:, :, :2]

colors = np.load(f"{DATA_DIR}/points3d_colors.npy")
n_visible_total = visibility.sum().item()
print(f"  Views: {n_views}, Points: {n_points}")
print(f"  Total visible observations: {n_visible_total:.0f}")

# ==================== Initialize Parameters ====================
focal = nn.Parameter(torch.tensor(900.0, dtype=torch.float32, device=DEVICE))

euler_angles = nn.Parameter(torch.zeros(n_views, 3, dtype=torch.float32, device=DEVICE))

translations = nn.Parameter(torch.zeros(n_views, 3, dtype=torch.float32, device=DEVICE))
with torch.no_grad():
    translations[:, 2] = -2.5

torch.manual_seed(42)
points_3d = nn.Parameter(torch.randn(n_points, 3, dtype=torch.float32, device=DEVICE) * 0.1)


# ==================== Projection ====================
def euler_to_rotation_matrix(euler):
    """Euler angles -> rotation matrices (XYZ convention: R = Rz @ Ry @ Rx).
    Input: (*, 3) -> Output: (*, 3, 3)
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    batch_shape = euler.shape[:-1]
    dev, dt = euler.device, euler.dtype

    # Rx
    cx, sx = torch.cos(rx), torch.sin(rx)
    Rx = torch.zeros(*batch_shape, 3, 3, device=dev, dtype=dt)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cx
    Rx[..., 1, 2] = -sx
    Rx[..., 2, 1] = sx
    Rx[..., 2, 2] = cx

    # Ry
    cy, sy = torch.cos(ry), torch.sin(ry)
    Ry = torch.zeros(*batch_shape, 3, 3, device=dev, dtype=dt)
    Ry[..., 0, 0] = cy
    Ry[..., 0, 2] = sy
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sy
    Ry[..., 2, 2] = cy

    # Rz
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rz = torch.zeros(*batch_shape, 3, 3, device=dev, dtype=dt)
    Rz[..., 0, 0] = cz
    Rz[..., 0, 1] = -sz
    Rz[..., 1, 0] = sz
    Rz[..., 1, 1] = cz
    Rz[..., 2, 2] = 1.0

    return Rz @ Ry @ Rx


def project(pts_3d, R, T, f):
    """Pinhole projection: 3D points -> 2D image coordinates.
    pts_3d: (N, 3), R: (V, 3, 3), T: (V, 3), f: scalar
    Returns: (V, N, 2)
    Formula: u = -f * Xc/Zc + cx, v = f * Yc/Zc + cy
    """
    Xc = torch.einsum('vab,nb->vna', R, pts_3d) + T.unsqueeze(1)
    Zc = Xc[..., 2]
    u = -f * Xc[..., 0] / Zc + CX
    v = f * Xc[..., 1] / Zc + CY
    return torch.stack([u, v], dim=-1)


# ==================== Optimization ====================
params = [
    {'params': [points_3d], 'lr': LR_POINTS},
    {'params': [euler_angles, translations], 'lr': LR_CAMERAS},
    {'params': [focal], 'lr': LR_FOCAL},
]
optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[500, 1000, 1500], gamma=0.5
)

losses = []
mae_pixels = []
print(f"\nStarting optimization ({N_ITERATIONS} iterations)...")
t_start = time.time()

for it in range(N_ITERATIONS):
    optimizer.zero_grad()

    R = euler_to_rotation_matrix(euler_angles)
    proj = project(points_3d, R, translations, focal)

    errors = proj - observed_2d
    squared_errors = (errors ** 2).sum(dim=-1)

    masked_se = squared_errors * visibility
    n_visible = visibility.sum()
    loss = masked_se.sum() / n_visible

    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    with torch.no_grad():
        abs_errors = torch.sqrt(squared_errors + 1e-8)
        mae = (abs_errors * visibility).sum() / n_visible
        mae_pixels.append(mae.item())

    if (it + 1) % 200 == 0:
        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]['lr']
        print(f"  Iter {it+1:4d}/{N_ITERATIONS} | "
              f"Loss: {loss.item():.4f} | MAE: {mae.item():.2f} px | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

t_total = time.time() - t_start
print(f"\nOptimization complete in {t_total:.1f}s")
print(f"  Final loss (MSE): {losses[-1]:.6f}")
print(f"  Final MAE: {mae_pixels[-1]:.2f} pixels")
print(f"  Final RMSE: {np.sqrt(losses[-1]):.2f} pixels")
print(f"  Optimized focal length: {focal.item():.2f}")

# ==================== Evaluate ====================
with torch.no_grad():
    R_final = euler_to_rotation_matrix(euler_angles)
    proj_final = project(points_3d, R_final, translations, focal)
    errors_final = (proj_final - observed_2d).norm(dim=-1)
    # Per-view statistics
    for v_idx in [0, 12, 25, 37, 49]:
        view_err = errors_final[v_idx][visibility[v_idx] > 0.5]
        print(f"  View {v_idx:02d}: MAE = {view_err.mean().item():.2f} px, "
              f"median = {view_err.median().item():.2f} px, "
              f"max = {view_err.max().item():.2f} px")

# ==================== Save Results ====================
# Loss curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses, linewidth=0.5, color='#1f77b4')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean Squared Reprojection Error (px²)')
axes[0].set_title('Bundle Adjustment Loss')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

zoom_start = max(0, len(losses) - 500)
axes[1].plot(range(zoom_start, len(losses)), losses[zoom_start:],
             linewidth=1, color='#1f77b4')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Mean Squared Reprojection Error (px²)')
axes[1].set_title(f'Loss (last {len(losses) - zoom_start} iterations)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
plt.close()
print(f"  Loss curve saved to {OUTPUT_DIR}/loss_curve.png")

# MAE curve (more interpretable)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mae_pixels, linewidth=0.5, color='#d62728')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Absolute Error (pixels)')
ax.set_title('Reprojection MAE over Optimization')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mae_curve.png", dpi=150)
plt.close()
print(f"  MAE curve saved to {OUTPUT_DIR}/mae_curve.png")

# OBJ file with per-vertex colors
pts_np = points_3d.detach().cpu().numpy()
focal_val = focal.item()

with open(f"{OUTPUT_DIR}/reconstructed.obj", 'w') as f:
    f.write("# Bundle Adjustment Reconstruction\n")
    f.write(f"# Points: {n_points}, Views: {n_views}\n")
    f.write(f"# Focal length: {focal_val:.2f}\n")
    f.write(f"# Final MSE: {losses[-1]:.6f}, MAE: {mae_pixels[-1]:.2f} px\n")
    for i in range(n_points):
        x, y, z = pts_np[i]
        r, g, b = colors[i]
        f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
print(f"  OBJ saved to {OUTPUT_DIR}/reconstructed.obj")

# Camera parameters
np.savez(
    f"{OUTPUT_DIR}/cameras.npz",
    focal=focal_val,
    euler_angles=euler_angles.detach().cpu().numpy(),
    translations=translations.detach().cpu().numpy(),
    losses=np.array(losses),
    mae_pixels=np.array(mae_pixels),
)
print(f"  Camera parameters saved to {OUTPUT_DIR}/cameras.npz")

print("\nDone! Results in output/")
