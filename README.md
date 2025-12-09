# PIV Project 2

Computer Vision project for image processing and 3D reconstruction.

## Features

- **SIFT Feature Matching**: Detect and match keypoints between two RGB images
- **Camera Pose Estimation**: Estimate rotation (R) and translation (t) using Essential Matrix with RANSAC
- **3D Point Cloud**: Triangulate matched points to create 3D reconstruction in camera-1 frame
- **Homography Computation**: Calculate homographies for image sequences relative to a reference image
- **Visualization**: Display 3D point clouds and warped image overlays

## Requirements

- Python 3.11
- NumPy
- Matplotlib
- OpenCV (cv2)
- Pillow

## Installation

Create and activate the conda environment:

```bash
conda create -n piv-project python=3.11 numpy matplotlib pillow opencv -y
conda activate piv-project
```

## Usage

```bash
python Home2.py
```

The script expects two input images:
- `320.jpg` (reference image)
- `330.jpg` (target image)

## Output

- Console output with match statistics, R/t estimates, and inlier counts
- Interactive matplotlib plots showing:
  - Original reference image
  - 3D point cloud
  - Homography visualizations
- Saved files:
  - `homography_visualization.png`
  - `homographies.npy`

## Project Structure

```
Project1/
├── Home2.py              # Main script
├── 320.jpg              # Reference image
├── 330.jpg              # Target image
├── .gitignore
└── README.md
```

## Algorithm Pipeline

1. Load two RGB images
2. Extract SIFT keypoints and match using ratio test
3. Estimate Essential Matrix with RANSAC
4. Recover camera pose (R, t) using recoverPose
5. Triangulate 3D points with cheirality check
6. Compute homographies for image alignment
7. Warp and blend images for visualization

## Authors

PIV Course - Universidade de Lisboa
