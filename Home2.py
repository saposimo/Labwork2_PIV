import numpy as np
import matplotlib.pyplot as plt
import cv2

# NOTE: Usare solo NumPy/SciPy per tutto il pipeline eccetto feature detection/matching (SIFT/BFMatcher).

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

################################################################################
# ESSENTIAL MATRIX, POSE, TRIANGULATION (NumPy only)
################################################################################

def _to_normalized(pts_px, K):
    """Convert pixel coords to normalized camera coords (homogeneous division)."""
    pts_h = np.column_stack([pts_px, np.ones(len(pts_px))])  # (N,3)
    K_inv = np.linalg.inv(K)
    pts_norm = (K_inv @ pts_h.T).T
    return pts_norm[:, :2]


def _eight_point_E(pts1_n, pts2_n):
    """Classic 8-point algorithm (normalized points)."""
    n = pts1_n.shape[0]
    A = np.zeros((n, 9))
    x1, y1 = pts1_n[:, 0], pts1_n[:, 1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1.0

    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    # Enforce rank-2 with singular values (1,1,0)
    U, S, Vt = np.linalg.svd(E)
    S = [1.0, 1.0, 0.0]
    E = U @ np.diag(S) @ Vt
    return E


def _sampson_error(E, pts1_n, pts2_n):
    """Sampson distance for essential matrix on normalized points."""
    x1 = np.column_stack([pts1_n, np.ones(len(pts1_n))])
    x2 = np.column_stack([pts2_n, np.ones(len(pts2_n))])

    Ex1 = E @ x1.T  # (3,N)
    Etx2 = E.T @ x2.T  # (3,N)
    x2tEx1 = np.sum(x2 * (E @ x1.T).T, axis=1)

    num = x2tEx1 ** 2
    denom = Ex1[0] ** 2 + Ex1[1] ** 2 + Etx2[0] ** 2 + Etx2[1] ** 2 + 1e-12
    return num / denom


def estimate_E_numpy(pts1_px, pts2_px, K, ransac_thresh=1e-3, max_iters=2000):
    """Estimate Essential matrix via 8-point + RANSAC using NumPy only."""
    if len(pts1_px) < 8:
        raise RuntimeError("Not enough points for E estimation")

    pts1_n = _to_normalized(pts1_px, K)
    pts2_n = _to_normalized(pts2_px, K)

    best_inliers = None
    best_E = None
    n = len(pts1_n)

    for _ in range(max_iters):
        idx = np.random.choice(n, 8, replace=False)
        E_candidate = _eight_point_E(pts1_n[idx], pts2_n[idx])
        err = _sampson_error(E_candidate, pts1_n, pts2_n)
        inliers = err < ransac_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E_candidate

    # Refine with all inliers
    if best_inliers is None or best_inliers.sum() < 8:
        raise RuntimeError("RANSAC failed to find a valid E")
    E_refined = _eight_point_E(pts1_n[best_inliers], pts2_n[best_inliers])
    return E_refined, best_inliers


def recover_pose_numpy(E, pts1_px, pts2_px, K):
    """Recover R,t from E using cheirality check (NumPy only)."""
    pts1_n = _to_normalized(pts1_px, K)
    pts2_n = _to_normalized(pts2_px, K)

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    candidates = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]

    def triangulate_points_numpy(pts1n, pts2n, R, t):
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t.reshape(3, 1)])
        pts_3d = []
        for (u1, v1), (u2, v2) in zip(pts1n, pts2n):
            A = np.array([
                u1 * P1[2] - P1[0],
                v1 * P1[2] - P1[1],
                u2 * P2[2] - P2[0],
                v2 * P2[2] - P2[1],
            ])
            _, _, Vt_tri = np.linalg.svd(A)
            X = Vt_tri[-1]
            X = X[:3] / X[3]
            pts_3d.append(X)
        return np.array(pts_3d)

    best_count = -1
    best_R, best_t = None, None

    for R_cand, t_cand in candidates:
        pts_3d = triangulate_points_numpy(pts1_n, pts2_n, R_cand, t_cand)
        if pts_3d.size == 0:
            continue
        z1 = pts_3d[:, 2]
        pts_cam2 = (R_cand @ pts_3d.T + t_cand.reshape(3, 1)).T
        z2 = pts_cam2[:, 2]
        positive = (z1 > 0) & (z2 > 0)
        count = positive.sum()
        if count > best_count:
            best_count = count
            best_R, best_t = R_cand, t_cand

    if best_R is None:
        raise RuntimeError("Failed to recover pose")

    # Keep only points in front of both cameras for downstream steps
    pts_3d_final = triangulate_points_numpy(pts1_n, pts2_n, best_R, best_t)
    z1 = pts_3d_final[:, 2]
    pts_cam2 = (best_R @ pts_3d_final.T + best_t.reshape(3, 1)).T
    z2 = pts_cam2[:, 2]
    cheirality_mask = (z1 > 0) & (z2 > 0)

    return best_R, best_t, cheirality_mask, pts_3d_final


def triangulate_points_numpy(pts1_px, pts2_px, K, R, t):
    """Triangulation in camera-1 frame using only NumPy."""
    pts1_n = _to_normalized(pts1_px, K)
    pts2_n = _to_normalized(pts2_px, K)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    pts_3d = []
    for (u1, v1), (u2, v2) in zip(pts1_n, pts2_n):
        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]
        pts_3d.append(X)
    pts_3d = np.array(pts_3d)

    z1 = pts_3d[:, 2]
    pts_cam2 = (R @ pts_3d.T + t.reshape(3, 1)).T
    z2 = pts_cam2[:, 2]
    mask = (z1 > 0) & (z2 > 0)
    return pts_3d[mask]


# --- Run estimation pipeline (NumPy) ---
E_est, inliers_E = estimate_E_numpy(pts1, pts2, K)
pts1_E = pts1[inliers_E]
pts2_E = pts2[inliers_E]

R_est, t_est, cheirality_mask, pts3d_full = recover_pose_numpy(E_est, pts1_E, pts2_E, K)
pts1_in = pts1_E[cheirality_mask]
pts2_in = pts2_E[cheirality_mask]
points_3d = pts3d_full[cheirality_mask]

print("Estimated R:\n", R_est)
print("Estimated t:\n", t_est)
print(f"Inliers after E-RANSAC: {len(pts1_E)}")
print(f"Cheirality-consistent points: {len(points_3d)}")

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

def normalize_points(points):
    """Hartley normalization for homography/8pt DLT."""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1))) + 1e-12
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.column_stack([points, np.ones(len(points))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def compute_homography_dlt(src_pts, dst_pts):
    """DLT homography from normalized points."""
    n = len(src_pts)
    A = []
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / (H[2, 2] + 1e-12)


def find_homography_ransac(src_pts, dst_pts, threshold=5.0, max_iterations=2000):
    """NumPy-only RANSAC for homography."""
    if len(src_pts) < 4:
        return None, None

    src_pts = np.asarray(src_pts, dtype=float)
    dst_pts = np.asarray(dst_pts, dtype=float)

    best_H, best_inliers = None, None
    N = len(src_pts)

    # Pre-normalize all points once
    src_norm_all, T_src = normalize_points(src_pts)
    dst_norm_all, T_dst = normalize_points(dst_pts)

    for _ in range(max_iterations):
        idx = np.random.choice(N, 4, replace=False)
        H_tilde = compute_homography_dlt(src_norm_all[idx], dst_norm_all[idx])
        H_candidate = np.linalg.inv(T_dst) @ H_tilde @ T_src

        # Reproject src -> dst
        src_h = np.column_stack([src_pts, np.ones(N)])
        proj = (H_candidate @ src_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-12)
        err = np.linalg.norm(proj - dst_pts, axis=1)
        inliers = err < threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_H = H_candidate

    if best_H is None:
        return None, None

    # Optional re-fit on inliers
    src_in = src_pts[best_inliers]
    dst_in = dst_pts[best_inliers]
    src_norm_in, T_src_in = normalize_points(src_in)
    dst_norm_in, T_dst_in = normalize_points(dst_in)
    H_tilde = compute_homography_dlt(src_norm_in, dst_norm_in)
    best_H = np.linalg.inv(T_dst_in) @ H_tilde @ T_src_in

    return best_H, best_inliers


def warp_image_numpy(src_img, H, output_shape):
    """Warp image using homography H (NumPy only, nearest-neighbor)."""
    h_out, w_out = output_shape
    h_src, w_src = src_img.shape[:2]

    u_dst, v_dst = np.meshgrid(np.arange(w_out), np.arange(h_out))
    coords_dst_h = np.stack([u_dst.flatten(), v_dst.flatten(), np.ones(u_dst.size)])

    H_inv = np.linalg.inv(H)
    coords_src_h = H_inv @ coords_dst_h
    coords_src_h /= (coords_src_h[2, :] + 1e-12)

    u_src = coords_src_h[0, :].astype(int)
    v_src = coords_src_h[1, :].astype(int)

    mask = (u_src >= 0) & (u_src < w_src) & (v_src >= 0) & (v_src < h_src)

    warped = np.zeros((h_out, w_out, src_img.shape[2]), dtype=src_img.dtype)
    dest_y = coords_dst_h[1, :].astype(int)[mask]
    dest_x = coords_dst_h[0, :].astype(int)[mask]
    warped[dest_y, dest_x] = src_img[v_src[mask], u_src[mask]]

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
        H_identity = np.eye(3)
        homographies[img_name] = H_identity
        inlier_counts[img_name] = len(pts1_in)
        print(f"{img_name} -> {ref_name}: Identity (reference)")
    else:
        # Homography 330 -> 320 using NumPy RANSAC
        H_330_to_320, mask_H = find_homography_ransac(
            pts2_in,  # src (330)
            pts1_in,  # dst (320)
            threshold=5.0,
            max_iterations=2000,
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
axes[0, 0].imshow(img1)
axes[0, 0].set_title("Reference: 320.jpg")
axes[0, 0].axis("off")

# (2) Immagine 330 originale
axes[0, 1].imshow(img2)
axes[0, 1].set_title("Original: 330.jpg")
axes[0, 1].axis("off")

# (3) 330 warppata nel frame di 320
warped_330 = None
if "330.jpg" in homographies:
    H_330_to_320 = homographies["330.jpg"]
    warped_330 = warp_image_numpy(img2_bgr, H_330_to_320, (H, W))
    axes[1, 0].imshow(warped_330[:, :, ::-1])  # BGR -> RGB
    axes[1, 0].set_title(f"330 warped -> 320 frame\n({inlier_counts['330.jpg']} inliers)")
else:
    axes[1, 0].text(0.5, 0.5, "Homography failed",
                    ha='center', va='center', fontsize=12)
axes[1, 0].axis("off")

# (4) Overlay tra riferimento e 330 warppata
if warped_330 is not None:
    ref_rgb = img1
    warped_rgb = warped_330[:, :, ::-1]  # BGR -> RGB
    alpha = 0.5
    overlay = (alpha * ref_rgb + (1 - alpha) * warped_rgb).astype(np.uint8)
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
