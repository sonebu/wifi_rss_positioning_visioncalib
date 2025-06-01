# spatial_error_budget.py
#
# Requirements: numpy, opencv-python (>4.5), matplotlib (for histograms)

import numpy as np, cv2, json, matplotlib.pyplot as plt

# ------------  USER-EDITABLE PARAMETERS  --------------------------
h_cam        = 3.0                      # camera height [m]
floor_h      = 25.0                      # vertical coverage of FoV [m]
sensor_h_px  = 1080                     # sensor vertical pixels
scale_m_per_px = floor_h / sensor_h_px  # ground-plane metres per pixel

sigma_px     = 0.5 * scale_m_per_px     # half-pixel centroiding

tag_sigma    = 0.30                     # 1-σ placement jitter [m]
n_trials     = 10000                    # Monte-Carlo iterations

det_sigma_px = 3.0                      # detector jitter [px]
sigma_det    = det_sigma_px * scale_m_per_px

foot_radius  = 0.20                     # max horizontal phone shift [m]
sigma_foot   = foot_radius / np.sqrt(6) # variance of uniform disc

# 4 nominal tag world-coords (square 4 m × 4 m)
tags_W = np.float32([
    [-8.0,  8.0, 0.0],
    [ 8.0,  8.0, 0.0],
    [ 8.0, -8.0, 0.0],
    [-8.0, -8.0, 0.0]
])

# synthetic camera intrinsics (fx≈fy, cx,cy centre):
f_px = 1000
K = np.array([[f_px, 0, sensor_h_px/2],
              [0, f_px, sensor_h_px/2],
              [0, 0, 1]], dtype=np.float64)

# -----------------------------------------------------------------
def one_trial():
    """Return one Monte-Carlo sample of σ_tag (metres)."""

    # 1) Jitter tags in world X,Y by N(0, tag_sigma^2)
    jitter       = np.random.randn(4, 3) * np.array([tag_sigma, tag_sigma, 0])
    tags_W_pert  = tags_W + jitter

    # 2) Project perturbed tags to image using the true pose
    rvec_true = np.zeros(3, dtype=np.float64)
    tvec_true = np.array([0, 0, -h_cam], dtype=np.float64)
    img_pts, _ = cv2.projectPoints(tags_W_pert, rvec_true, tvec_true, K, None)

    # 3) Solve PnP \{~nominal tags + noisy img points~\}
    ok, rvec_est, tvec_est = cv2.solvePnP(tags_W, img_pts, K, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0

    # 4) Re-project nominal tags with the *estimated* pose
    img_reproj, _ = cv2.projectPoints(tags_W, rvec_est, tvec_est, K, None)

    # 5) Pixel reprojection error (RMS over four tags)
    err_px = np.linalg.norm(img_pts.reshape(-1, 2) -
                            img_reproj.reshape(-1, 2), axis=1).mean()

    # 6) Convert pixel error to metres via scale
    return err_px * scale_m_per_px


errs = [one_trial() for _ in range(n_trials)]
sigma_tag = np.std(errs)

sigma_t = np.sqrt(sigma_px**2 + sigma_tag**2 + sigma_det**2 + sigma_foot**2)

print(json.dumps({
    "sigma_px"  : sigma_px,
    "sigma_tag" : sigma_tag,
    "sigma_det" : sigma_det,
    "sigma_foot": sigma_foot,
    "sigma_t"   : sigma_t
}, indent=2))

# quick sanity-check plot
plt.hist(errs, bins=40, alpha=0.7)
plt.xlabel("Tag-placement reprojection error [m]")
plt.ylabel("count")
plt.title("Monte-Carlo σ_tag ≈ %.3f m" % sigma_tag)
plt.tight_layout()
plt.savefig("mc_tag_hist.png", dpi=300)
