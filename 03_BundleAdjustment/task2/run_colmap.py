"""COLMAP 3D reconstruction pipeline using pycolmap Python bindings."""

import os
import shutil
import pycolmap

# ==================== Configuration ====================
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
COLMAP_DIR = os.path.join(DATA_DIR, "colmap")
DATABASE_PATH = os.path.join(COLMAP_DIR, "database.db")
OUTPUT_PATH = os.path.join(COLMAP_DIR, "sparse")

# Clean previous results
if os.path.exists(COLMAP_DIR):
    shutil.rmtree(COLMAP_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==================== Step 1: Feature Extraction ====================
print("=== Step 1: Feature Extraction ===")
reader_opts = pycolmap.ImageReaderOptions()
reader_opts.camera_model = "PINHOLE"

sift_opts = pycolmap.SiftExtractionOptions()
sift_opts.max_num_features = 8192

pycolmap.extract_features(
    database_path=DATABASE_PATH,
    image_path=IMAGE_DIR,
    camera_mode=pycolmap.CameraMode.SINGLE,
    camera_model="PINHOLE",
    reader_options=reader_opts,
    sift_options=sift_opts,
    device=pycolmap.Device.auto,
)
print("  Done.")

# ==================== Step 2: Feature Matching ====================
print("\n=== Step 2: Feature Matching ===")
match_opts = pycolmap.SiftMatchingOptions()
pycolmap.match_exhaustive(
    database_path=DATABASE_PATH,
    sift_options=match_opts,
)
print("  Done.")

# ==================== Step 3: Sparse Reconstruction ====================
print("\n=== Step 3: Sparse Reconstruction (Bundle Adjustment) ===")
mapper_opts = pycolmap.IncrementalPipelineOptions()

reconstructions = pycolmap.incremental_mapping(
    database_path=DATABASE_PATH,
    image_path=IMAGE_DIR,
    output_path=OUTPUT_PATH,
    options=mapper_opts,
)
print(f"  Reconstructions found: {len(reconstructions)}")

if not reconstructions:
    print("\n[WARNING] No reconstruction found.")
    print("The rendered images might not have enough texture for SIFT keypoints.")
else:
    for idx, rec in reconstructions.items() if isinstance(reconstructions, dict) else enumerate(reconstructions):
        if isinstance(reconstructions, dict):
            rec = reconstructions[idx]
        rec_path = os.path.join(OUTPUT_PATH, str(idx))
        rec.write(rec_path)
        print(f"\n  Model {idx}:")
        print(f"    Registered images: {rec.num_reg_images()}")
        print(f"    3D points: {rec.num_points3D()}")

        if rec.num_reg_images() > 0:
            image_ids = list(rec.images.keys())
            for img_id in image_ids[:3]:
                img = rec.images[img_id]
                cam = rec.cameras[img.camera_id]
                print(f"    Camera: focal={cam.mean_focal_length:.1f}")

        print(f"    Results: {rec_path}")

print("\n=== Done! ===")
