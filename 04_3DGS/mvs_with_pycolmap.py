"""Run COLMAP SfM using pycolmap Python bindings (replaces CLI-based mvs_with_colmap.py).

Usage:
    python mvs_with_pycolmap.py --data_dir data/chair
"""

import os
import argparse
import pycolmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run COLMAP SfM via pycolmap')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to input directory containing images in data_dir/images')
    args = parser.parse_args()
    data_dir = args.data_dir

    image_dir = os.path.join(data_dir, 'images')
    database_path = os.path.join(data_dir, 'database.db')
    sparse_dir = os.path.join(data_dir, 'sparse')
    text_dir = os.path.join(sparse_dir, '0_text')

    # 1. Feature extraction (SIFT)
    print('Step 1/4: Extracting SIFT features...')
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.use_gpu = False  # CPU-only
    sift_options.num_threads = 4  # Limit threads to avoid OOM
    sift_options.max_image_size = 800  # Process at native resolution
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = 'PINHOLE'
    reader_options.default_focal_length_factor = 1.2
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model='PINHOLE',
        reader_options=reader_options,
        sift_options=sift_options,
        device=pycolmap.Device.cpu,
    )
    print('Feature extraction done.')

    # 2. Exhaustive matching
    print('Step 2/4: Exhaustive matching...')
    match_options = pycolmap.SiftMatchingOptions()
    match_options.use_gpu = False
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=match_options,
    )
    print('Matching done.')

    # 3. Incremental mapping (sparse reconstruction)
    print('Step 3/4: Sparse reconstruction...')
    os.makedirs(sparse_dir, exist_ok=True)
    output_path = os.path.join(sparse_dir, '0')
    os.makedirs(output_path, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=output_path,
    )
    print(f'Got {len(reconstructions)} reconstruction(s).')

    if not reconstructions:
        print('ERROR: No reconstructions produced!')
        import sys
        sys.exit(1)

    # 4. Find the largest reconstruction and export as text
    print('Step 4/4: Exporting to text format...')
    best = max(reconstructions.values(), key=lambda r: r.num_points3D())
    print(f'Best reconstruction: {best.num_reg_images()} images, '
          f'{best.num_points3D()} 3D points')

    os.makedirs(text_dir, exist_ok=True)
    best.write_text(text_dir)
    print(f'Sparse 3D reconstruction saved in: {text_dir}')
    print('COLMAP SfM pipeline completed successfully!')
