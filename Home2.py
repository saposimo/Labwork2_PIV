"""
Lab Work 2 – NUMPY-ONLY GEOMETRY PIPELINE
------------------------------------------
Allowed:    SIFT, BFMatcher from OpenCV
Forbidden:  All OpenCV geometry (Essential, Pose, Triangulation, Homography, Warp)
Everything else must be implemented with NumPy/SciPy.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CONFIGURATION (EDITABLE)
# ==============================================================================

CONFIG = {
    "ref_name": "lisbon_0001.jpg",     # reference image
    "prefix": "lisbon_",               # load only prefixXXX.jpg
    "max_images": 2,                  # max number of images to load
    "fx_fy": 1180.0,                   # camera intrinsics guess
    "ransac_E_thresh": 1e-3,           # RANSAC threshold for Essential
    "ransac_H_thresh": 4.0,            # RANSAC threshold for homography
    "ransac_iters": 2000,
}

# ==============================================================================
# IMAGE LOADING
# ==============================================================================

def load_image_sequence(ref_name, prefix, max_images):
    """Load reference first, then all other prefix*.jpg images."""
    files = sorted(Path(".").glob(f"{prefix}*.jpg"))
    seq = []
    for f in files:
        name = f.name
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if name == ref_name:
            seq.insert(0, (img, name))
        else:
            seq.append((img, name))
        if len(seq) >= max_images:
            break

    if len(seq) == 0 or seq[0][1] != ref_name:
        raise FileNotFoundError(f"Reference '{ref_name}' not found.")

    return seq


# ==============================================================================
# FEATURE MATCHING (allowed)
# ==============================================================================

def sift_match(img1, img2, ratio=0.75):
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(d1, d2, k=2)

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


# ==============================================================================
# CAMERA NORMALIZATION
# ==============================================================================

def to_normalized(pts, K):
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    Kinv = np.linalg.inv(K)
    p = (Kinv @ pts_h.T).T
    return p[:, :2]


# ==============================================================================
# ESSENTIAL MATRIX  - eight-point + RANSAC (NUMPY)
# ==============================================================================

def eight_point_E(pts1, pts2):
    """8pt algorithm (after normalization). Enforce rank-2."""
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(len(pts1))
    ])

    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    # rank-2 enforcement
    U, _, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    return U @ S @ Vt


def sampson_error(E, pts1, pts2):
    """Sampson distance for essential."""
    x1 = np.column_stack([pts1, np.ones(len(pts1))])
    x2 = np.column_stack([pts2, np.ones(len(pts2))])

    Ex1 = E @ x1.T
    Etx2 = E.T @ x2.T
    x2tEx1 = np.sum(x2 * (E @ x1.T).T, axis=1)

    num = x2tEx1 ** 2
    den = Ex1[0]**2 + Ex1[1]**2 + Etx2[0]**2 + Etx2[1]**2 + 1e-12
    return num / den


def estimate_E_ransac(pts1, pts2, K, thresh, iters):
    """NumPy-only RANSAC for Essential E."""
    pts1n = to_normalized(pts1, K)
    pts2n = to_normalized(pts2, K)
    N = len(pts1)

    best_E, best_inliers = None, np.zeros(N, bool)

    for _ in range(iters):
        idx = np.random.choice(N, 8, replace=False)
        E_candidate = eight_point_E(pts1n[idx], pts2n[idx])
        err = sampson_error(E_candidate, pts1n, pts2n)
        inl = err < thresh

        if inl.sum() > best_inliers.sum():
            best_inliers = inl
            best_E = E_candidate

    if best_E is None:
        raise RuntimeError("RANSAC failed for Essential.")

    # refine
    E_ref = eight_point_E(pts1n[best_inliers], pts2n[best_inliers])
    return E_ref, best_inliers


# ==============================================================================
# POSE RECOVERY FROM ESSENTIAL (NUMPY)
# ==============================================================================

def recover_pose(E, pts1, pts2, K):
    pts1n = to_normalized(pts1, K)
    pts2n = to_normalized(pts2, K)

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    candidates = [
        (U @ W @ Vt,  U[:,2]),
        (U @ W @ Vt, -U[:,2]),
        (U @ W.T @ Vt,  U[:,2]),
        (U @ W.T @ Vt, -U[:,2]),
    ]

    def triangulate(R,t):
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = np.hstack([R, t.reshape(3,1)])
        pts3d = []
        for (u1,v1),(u2,v2) in zip(pts1n, pts2n):
            A = np.array([
                u1*P1[2]-P1[0],
                v1*P1[2]-P1[1],
                u2*P2[2]-P2[0],
                v2*P2[2]-P2[1]
            ])
            _,_,VtA = np.linalg.svd(A)
            X = VtA[-1]
            X = X[:3]/X[3]
            pts3d.append(X)
        return np.array(pts3d)

    best_R, best_t, best_count = None, None, -1

    for R,t in candidates:
        pts = triangulate(R,t)
        z1 = pts[:,2]
        z2 = (R @ pts.T + t.reshape(3,1))[2]
        in_front = (z1>0) & (z2>0)
        if in_front.sum() > best_count:
            best_R, best_t, best_count = R, t, in_front.sum()

    return best_R, best_t


def triangulate_points(pts1, pts2, K, R, t):
    """Triangulate all inliers (NumPy)."""
    pts1n = to_normalized(pts1, K)
    pts2n = to_normalized(pts2, K)

    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([R, t.reshape(3,1)])

    out = []
    for (u1,v1),(u2,v2) in zip(pts1n, pts2n):
        A = np.array([
            u1*P1[2]-P1[0],
            v1*P1[2]-P1[1],
            u2*P2[2]-P2[0],
            v2*P2[2]-P2[1],
        ])
        _,_,Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3]/X[3]
        out.append(X)
    return np.array(out)


# ==============================================================================
# HOMOGRAPHY (DLT + RANSAC, NUMPY)
# ==============================================================================

def normalize_h(points):
    c = np.mean(points, axis=0)
    s = np.sqrt(2) / (np.mean(np.linalg.norm(points - c, axis=1)) + 1e-12)

    T = np.array([
        [s,0,-s*c[0]],
        [0,s,-s*c[1]],
        [0,0,1],
    ])
    pts_h = np.column_stack([points, np.ones(len(points))])
    ptsn = (T @ pts_h.T).T
    return ptsn[:,:2], T


def H_from_dlt(p1, p2):
    A = []
    for (x,y),(xp,yp) in zip(p1,p2):
        A.append([-x,-y,-1,0,0,0,x*xp,y*xp,xp])
        A.append([0,0,0,-x,-y,-1,x*yp,y*yp,yp])
    _,_,Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3,3)
    return H / (H[2,2]+1e-12)


def ransac_homography(pts_src, pts_dst, thresh, iters):
    N = len(pts_src)
    best_H, best_inl = None, None

    p1n, T1 = normalize_h(pts_src)
    p2n, T2 = normalize_h(pts_dst)

    for _ in range(iters):
        idx = np.random.choice(N, 4, False)
        Htilde = H_from_dlt(p1n[idx], p2n[idx])
        Hcand = np.linalg.inv(T2) @ Htilde @ T1

        src_h = np.column_stack([pts_src, np.ones(N)])
        proj = (Hcand @ src_h.T).T
        proj = proj[:,:2]/(proj[:,2:3]+1e-12)

        err = np.linalg.norm(proj - pts_dst, axis=1)
        inl = err < thresh

        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
            best_H = Hcand

    return best_H, best_inl


# ==============================================================================
# WARPING (NUMPY)
# ==============================================================================

def warp_numpy(img, H, out_wh):
    h_out, w_out = out_wh
    h, w = img.shape[:2]

    u_dst, v_dst = np.meshgrid(np.arange(w_out), np.arange(h_out))
    pts = np.stack([u_dst.ravel(), v_dst.ravel(), np.ones(u_dst.size)])

    Hinv = np.linalg.inv(H)
    src = Hinv @ pts
    src /= src[2] + 1e-12
    u, v = src[0].astype(int), src[1].astype(int)

    mask = (u>=0)&(u<w)&(v>=0)&(v<h)
    out = np.zeros((h_out,w_out,3), img.dtype)
    out[v_dst.ravel()[mask], u_dst.ravel()[mask]] = img[v[mask], u[mask]]
    return out

def blend_numpy(base_img, new_img, alpha=0.5):
    """
    Fonde new_img dentro base_img solo dove new_img è valido (non nero).
    Tutto in NumPy, niente OpenCV.
    base_img, new_img: BGR, stessa shape.
    """
    out = base_img.copy()
    # pixel validi: non tutti e tre i canali a zero
    mask = np.any(new_img != 0, axis=2)

    # blending solo sulla maschera
    base = base_img.astype(np.float32)
    new = new_img.astype(np.float32)

    out[mask] = (alpha * base[mask] + (1.0 - alpha) * new[mask]).astype(np.uint8)
    return out


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(config=CONFIG):
    print("\nLoading image sequence...")
    seq = load_image_sequence(config["ref_name"], config["prefix"], config["max_images"])

    # intrinsics
    first_img = seq[0][0]
    H_img, W_img = first_img.shape[:2]
    fx = fy = config["fx_fy"]
    cx, cy = W_img/2, H_img/2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    print("K =\n", K)

    # ----------------------------------------------------------
    # E / Pose / Triangulation (use first two images only)
    # ----------------------------------------------------------
    print("\n--- SIFT Matching (ref <-> next) ---")
    pts1, pts2 = sift_match(seq[0][0], seq[1][0])
    print(f"Matches: {len(pts1)}")

    print("\n--- Essential via RANSAC (NumPy) ---")
    E, inlE = estimate_E_ransac(
        pts1, pts2,
        K,
        config["ransac_E_thresh"],
        config["ransac_iters"]
    )
    pts1E, pts2E = pts1[inlE], pts2[inlE]
    print(f"Inliers (E): {inlE.sum()}")

    print("\n--- Recover Pose (NumPy) ---")
    R, t = recover_pose(E, pts1E, pts2E, K)
    print("R=\n", R)
    print("t=\n", t)

    print("\n--- Triangulate (NumPy) ---")
    pts3d = triangulate_points(pts1E, pts2E, K, R, t)
    print(f"3D points: {len(pts3d)}")

        # ----------------------------------------------------------
    # Homographies for entire sequence
    # ----------------------------------------------------------
    print("\n--- Homographies to reference ---")
    ref_img_bgr = seq[0][0]
    ref_name = seq[0][1]

    homos = {ref_name: np.eye(3)}
    inliers_H = {ref_name: len(pts1E)}

    for img, name in seq[1:]:
        pts_ref, pts_tgt = sift_match(ref_img_bgr, img)
        Hmat, inl = ransac_homography(
            pts_tgt, pts_ref,
            config["ransac_H_thresh"],
            config["ransac_iters"]
        )
        if Hmat is not None:
            homos[name] = Hmat
            inliers_H[name] = int(inl.sum())
            print(f"{name} -> {ref_name} : {inliers_H[name]} inliers")
        else:
            print(f"Failed for {name}")

    # ----------------------------------------------------------
    # Visualization – second image (reference + warped + overlay)
    # ----------------------------------------------------------
    canvas_hw = (H_img, W_img)   # (height, width)
    second_img_bgr, second_name = seq[1]

    print("\nWarping second image into reference frame...")

    warped2_bgr = warp_numpy(second_img_bgr, homos[second_name], canvas_hw)
    overlay2_bgr = blend_numpy(ref_img_bgr, warped2_bgr, alpha=0.5)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    # 1) Reference
    ax[0].imshow(ref_img_bgr[:, :, ::-1])  # BGR -> RGB
    ax[0].set_title("Reference")
    ax[0].axis("off")

    # 2) Original second
    ax[1].imshow(second_img_bgr[:, :, ::-1])
    ax[1].set_title("Original")
    ax[1].axis("off")

    # 3) Warped -> reference
    ax[2].imshow(warped2_bgr[:, :, ::-1])
    ax[2].set_title("Warped -> Reference Frame")
    ax[2].axis("off")

    # 4) Overlay reference + warped
    ax[3].imshow(overlay2_bgr[:, :, ::-1])
    ax[3].set_title("Overlay (Ref + Warped)")
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig("warp_overlay_second.png", dpi=120, bbox_inches="tight")
    print("Saved warp/overlay figure: warp_overlay_second.png")
    plt.show()

    # ----------------------------------------------------------
    # PANORAMICA: tutte le immagini warpate nel frame della reference
    # ----------------------------------------------------------
    print("\nBuilding panorama (all images -> reference frame)...")

    panorama_bgr = ref_img_bgr.copy()
    for img_bgr, name in seq[1:]:
        Hmat = homos.get(name, None)
        if Hmat is None:
            continue
        warped_bgr = warp_numpy(img_bgr, Hmat, canvas_hw)
        panorama_bgr = blend_numpy(panorama_bgr, warped_bgr, alpha=0.5)

    plt.figure(figsize=(10, 5))
    plt.imshow(panorama_bgr[:, :, ::-1])
    plt.title("Panorama in Reference Frame")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("panorama_reference_frame.png", dpi=120, bbox_inches="tight")
    print("Saved panorama: panorama_reference_frame.png")
    plt.show()

    return {
        "K": K,
        "R": R,
        "t": t,
        "E": E,
        "points3D": pts3d,
        "homographies": homos,
        "inliers_E": int(inlE.sum()),
        "inliers_H": inliers_H,
    }



# Run pipeline
if __name__ == "__main__":
    results = run_pipeline(CONFIG)
    np.save("lab2_results.npy", results)
    print("\nSaved results to lab2_results.npy")
