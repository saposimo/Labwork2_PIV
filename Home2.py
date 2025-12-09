import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==============================================================================
# LOAD IMAGES
# ==============================================================================

# Images (BGR for OpenCV)
img1_bgr = cv2.imread("320.jpg", cv2.IMREAD_COLOR)
img2_bgr = cv2.imread("330.jpg", cv2.IMREAD_COLOR)

if img1_bgr is None or img2_bgr is None:
    raise FileNotFoundError("Assicurati che 320.jpg e 330.jpg siano nella cartella corrente")

# Convert to RGB for plotting (solo per matplotlib)
img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

# Camera intrinsics (stima)
H, W = img1.shape[:2]
fx = fy = 1180.0
cx = W / 2.0
cy = H / 2.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

print(f"Image size: {W}x{H}")
print("K =\n", K)

# ==============================================================================
# SIFT + MATCHES
# ==============================================================================

def sift_keypoints_and_matches(im1, im2, ratio=0.75):
    """
    Trova keypoints SIFT e matcha i descrittori tra im1 e im2.
    Ritorna:
        pts1: punti 2D in im1
        pts2: punti 2D in im2
        good: lista di match buoni
        kp1, kp2: keypoints completi
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, good, kp1, kp2


pts1, pts2, good_matches, kp1, kp2 = sift_keypoints_and_matches(img1_bgr, img2_bgr)
print(f"SIFT matches (ratio-tested): {len(good_matches)}")

# ==============================================================================
# ESSENTIAL MATRIX + POSE (R, t)
# ==============================================================================

def estimate_pose_from_essential(pts1, pts2, K, ransac_thresh=1.0, prob=0.999):
    """
    Stima matrice essenziale E con RANSAC e poi recupera R,t con recoverPose.
    pts1, pts2: corrispondenze 2D (in pixel) tra img1 e img2.
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC,
                                   threshold=ransac_thresh,
                                   prob=prob)
    if E is None:
        raise RuntimeError("findEssentialMat failed")

    inliers_mask = mask.ravel().astype(bool)
    pts1_in = pts1[inliers_mask]
    pts2_in = pts2[inliers_mask]

    _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
    pose_inliers = mask_pose.ravel().astype(bool)
    pts1_pose = pts1_in[pose_inliers]
    pts2_pose = pts2_in[pose_inliers]

    return R, t.reshape(3), pts1_pose, pts2_pose


R_est, t_est, pts1_in, pts2_in = estimate_pose_from_essential(pts1, pts2, K)
print("Estimated R:\n", R_est)
print("Estimated t:\n", t_est)
print(f"Inliers after pose recovery: {len(pts1_in)}")

# ==============================================================================
# TRIANGULATION: 3D POINT CLOUD IN CAMERA-1 FRAME
# ==============================================================================

def triangulate_points(pts1, pts2, K, R, t):
    """
    Triangola punti corrispondenti pts1, pts2 usando le pose:
    Cam1: [I | 0]
    Cam2: [R | t]
    Ritorna punti 3D nel frame della camera 1.
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    pts1_h = pts1.T
    pts2_h = pts2.T
    homog = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts_3d = (homog[:3] / homog[3]).T  # (N,3)

    # Cheirality: tieni solo punti davanti ad entrambe le camere
    z1 = pts_3d[:, 2]
    pts_cam2 = (R @ pts_3d.T + t.reshape(3, 1)).T
    z2 = pts_cam2[:, 2]
    mask = (z1 > 0) & (z2 > 0)
    return pts_3d[mask]


points_3d = triangulate_points(pts1_in, pts2_in, K, R_est, t_est)
print(f"3D points kept (cheirality): {points_3d.shape[0]}")

# Visualizzazione rapida della nuvola 3D
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img1)
ax1.set_title("Image 1 (RGB)")
ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            s=1, c=points_3d[:, 2], cmap='viridis')
ax2.set_title("Triangulated 3D (cam1 frame)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
plt.tight_layout()
plt.show()

# ==============================================================================
# HOMOGRAPHY COMPUTATION FOR IMAGE SEQUENCE
# ==============================================================================

def compute_homography_from_matches(src_pts, dst_pts,
                                    method=cv2.RANSAC,
                                    ransac_thresh=2.0):
    """
    Calcola H tale che:
        dst_pts ~ H * src_pts
    (coordinate omogenee).
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Not enough points for homography")
        return None, None

    H, mask = cv2.findHomography(src_pts, dst_pts,
                                 method=method,
                                 ransacReprojThreshold=ransac_thresh)
    return H, mask


def warp_image(src_img, H, canvas_size):
    """
    Warpa src_img usando l'omografia H verso un canvas di dimensione canvas_size.
    canvas_size: (width, height)
    """
    width, height = canvas_size
    warped = cv2.warpPerspective(src_img, H, (width, height))
    return warped


# ------------------------------------------------------------------------------
# Sequence (per ora solo 320 e 330)
# ------------------------------------------------------------------------------

image_sequence = [
    (img1_bgr, "320.jpg"),
    (img2_bgr, "330.jpg"),
]

ref_name = "320.jpg"
ref_img = img1_bgr
print("\n--- Homography Sequence Processing ---")
print(f"Reference image: {ref_name}")

homographies = {}
inlier_counts = {}

for img, img_name in image_sequence:
    if img_name == ref_name:
        # Reference: identità
        H_identity = np.eye(3)
        homographies[img_name] = H_identity
        inlier_counts[img_name] = len(pts1_in)
        print(f"{img_name} -> {ref_name}: Identity (reference)")
    else:
        # Vogliamo l'omografia:  img_name (330)  --> reference (320)
        # Quindi src_pts = pts2_in (330), dst_pts = pts1_in (320)
        H_330_to_320, mask_H = compute_homography_from_matches(
            pts2_in,  # src (immagine 330)
            pts1_in,  # dst (immagine 320)
            method=cv2.RANSAC,
            ransac_thresh=2.0
        )

        if H_330_to_320 is not None:
            homographies[img_name] = H_330_to_320
            n_inliers = int(mask_H.sum()) if mask_H is not None else len(pts1_in)
            inlier_counts[img_name] = n_inliers
            print(f"{img_name} -> {ref_name}: H computed ({n_inliers} inliers)")
            print("H =\n", H_330_to_320, "\n")
        else:
            print(f"Failed to compute homography for {img_name}")

# ==============================================================================
# VISUALIZATION: WARP + OVERLAY
# ==============================================================================

# Canvas = stessa dimensione della reference
canvas_size = (W, H)  # (width, height)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Immagine di riferimento
axes[0, 0].imshow(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Reference: 320.jpg")
axes[0, 0].axis("off")

# (2) Immagine 330 originale
axes[0, 1].imshow(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Original: 330.jpg")
axes[0, 1].axis("off")

# (3) 330 warppata nel frame di 320
if "330.jpg" in homographies:
    H_330_to_320 = homographies["330.jpg"]
    warped_330 = warp_image(img2_bgr, H_330_to_320, canvas_size)
    axes[1, 0].imshow(cv2.cvtColor(warped_330, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"330 warped -> 320 frame\n({inlier_counts['330.jpg']} inliers)")
else:
    axes[1, 0].text(0.5, 0.5, "Homography failed",
                    ha='center', va='center', fontsize=12)
    warped_330 = None
axes[1, 0].axis("off")

# (4) Overlay tra riferimento e 330 warppata
if warped_330 is not None:
    ref_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped_330, cv2.COLOR_BGR2RGB)

    # semplice blend alpha
    alpha = 0.5
    overlay = cv2.addWeighted(ref_rgb, alpha, warped_rgb, 1 - alpha, 0)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay: 320 + warped 330")
else:
    axes[1, 1].text(0.5, 0.5, "No overlay", ha='center', va='center', fontsize=12)
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig('homography_visualization.png', dpi=100, bbox_inches='tight')
print("\nHomography visualization saved to: homography_visualization.png")
plt.show()

# ==============================================================================
# SAVE HOMOGRAPHIES
# ==============================================================================

homography_data = {
    'reference': ref_name,
    'homographies': homographies,
    'inlier_counts': inlier_counts,
}
np.save('homographies.npy', homography_data)
print("Homographies saved to: homographies.npy")
