"""
Shared constants and signal functions for the SIFT scale-space video (pure numpy, no Manim).
Importable by Scene 2 and Scene 3.
"""
import math

import numpy as np

F = 1.5                      # focal length
L_SET = 2.0                  # object length
Z_SET = [2, 4, 8]            # depths of the three Scene-2 signals
BG = 0.1                     # background brightness
PEAK = 1.0                   # object brightness

_erf = np.vectorize(math.erf, otypes=[float])


def widths():
    """Image widths F*L/Z for Z in Z_SET: [1.5, 0.75, 0.375]."""
    return [F * L_SET / Z for Z in Z_SET]


def box(x, w):
    """BG + (PEAK-BG) * 1[|x| <= w/2]."""
    x = np.asarray(x, dtype=float)
    return BG + (PEAK - BG) * (np.abs(x) <= w / 2 + 1e-12)


def gauss(x, sigma):
    """Unit-area Gaussian kernel."""
    x = np.asarray(x, dtype=float)
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def blurred_box(x, w, sigma):
    """Exact continuous convolution of `box` with a unit-area Gaussian of std sigma."""
    x = np.asarray(x, dtype=float)
    s = np.sqrt(2) * sigma
    return BG + (PEAK - BG) * 0.5 * (_erf((x + w / 2) / s) - _erf((x - w / 2) / s))


def peak_value(w, sigma):
    """Height of the blurred box at x = 0 (its maximum)."""
    return float(blurred_box(0.0, w, sigma))


# ----------------------------------------------------------------- Scene 4: heat on a rod
ROD_CENTERS = [-2.5, 0.0, 2.5]      # boxes for Z = 2, 4, 8 (widths 1.5, 0.75, 0.375)


def _boxes():
    return list(zip(widths(), ROD_CENTERS))


def rod_profile(x):
    """BG + sum of the three boxes (each box rises (PEAK-BG) above BG)."""
    x = np.asarray(x, dtype=float)
    return BG + sum(box(x - c, w) - BG for w, c in _boxes())


def rod_blurred(x, t):
    """Exact heat-equation solution u_t = u_xx from rod_profile: blur with sigma = sqrt(2t)."""
    x = np.asarray(x, dtype=float)
    if t <= 0:
        return rod_profile(x)
    sig = math.sqrt(2 * t)
    return BG + sum(blurred_box(x - c, w, sig) - BG for w, c in _boxes())


def _cell_average_profile(x, dx):
    """rod_profile averaged over the cell [x-dx/2, x+dx/2] (exact; keeps box edges off-grid errors O(dx^2))."""
    u = np.full_like(x, BG)
    for w, c in _boxes():
        lo = np.maximum(x - dx / 2, c - w / 2)
        hi = np.minimum(x + dx / 2, c + w / 2)
        u += (PEAK - BG) * np.clip(hi - lo, 0, None) / dx
    return u


def heat_ftcs(t_list, dx=0.01, L=10.0):
    """Explicit FTCS solver for u_t = u_xx on [-L, L], Dirichlet u = BG at both ends.

    Initial condition: rod_profile (cell-averaged on the grid). dt = 0.4 dx^2 (stable: dt <= dx^2/2);
    the last step to each requested time is shortened so snapshots land exactly on t.
    Returns (x, snapshots) with snapshots[i] = u at t_list[i] (in the order given).
    """
    x = np.arange(-L, L + dx / 2, dx)
    u = _cell_average_profile(x, dx)
    u[0] = u[-1] = BG
    dt = 0.4 * dx**2
    order = np.argsort(t_list)
    snaps = [None] * len(t_list)
    t_now = 0.0
    for i in order:
        target = float(t_list[i])
        n = int(math.floor((target - t_now) / dt + 1e-9))
        for _ in range(n):
            u[1:-1] += (dt / dx**2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        rem = target - t_now - n * dt
        if rem > 1e-15:
            u[1:-1] += (rem / dx**2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        t_now = target
        snaps[i] = u.copy()
    return x, np.array(snaps)


def half_life(w):
    """Diffusion time t at which a single box's center contrast halves: erf(w/(2 sqrt2 sigma)) = 1/2."""
    lo, hi = 1e-9, 100.0 * w
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(w / (2 * math.sqrt(2) * mid)) > 0.5:
            lo = mid            # still above half -> need larger sigma
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma**2 / 2


# ----------------------------------------------------------------- Scenes 7-9: 2D scale space
# Images are sampled on a pixel grid centered on the scene origin: column j, row i map to
# x = (j + 0.5 - W/2) / px, y = (H/2 - i - 0.5) / px  (px = pixels per scene unit, row 0 at the top).
SIFT_K = 2 ** (1 / 3)               # Lowe 2004: 3 intervals per octave
LOWE_CONTRAST = 0.03                # |D| below this is discarded (intensities in [0, 1])
LOWE_EDGE_R = 10.0                  # principal-curvature ratio limit

# Scenes 7-8 test image: one bright disk seen from 4Z, 2Z, Z (radii r, 2r, 4r), 12 x 4.5 units
CIRCLE_R0 = 0.25
CIRCLE_DISKS = [(-4.4, 0.0, CIRCLE_R0, PEAK - BG), (-2.0, 0.0, 2 * CIRCLE_R0, PEAK - BG),
                (2.6, 0.0, 4 * CIRCLE_R0, PEAK - BG)]          # (cx, cy, r, contrast)
CIRCLE_PX = 40                      # pixels per scene unit
CIRCLE_SHAPE = (181, 481)           # odd sizes: the scene origin falls on a pixel center
CIRCLE_SIGMA0 = CIRCLE_R0 / 2       # DoG layer i spans [k^i, k^(i+1)] * sigma0
CIRCLE_LAYERS = 9
# Scene 8 clutter: a faint disk (fails the contrast test) and a thin bright ellipse (its flanks fail the edge test)
CLUTTER_DISKS = [(0.3, -1.4, 0.4, 0.1)]
CLUTTER_ELLIPSES = [(0.3, 0.6, 1.0, 0.15, 75, PEAK - BG)]   # (cx, cy, a, b, angle_deg, contrast)


def pixel_grid(shape, px):
    """Scene-unit coordinates (X, Y) of the pixel centers of an image of `shape` = (H, W)."""
    h, w = shape
    xs = (np.arange(w) + 0.5 - w / 2) / px
    ys = (h / 2 - np.arange(h) - 0.5) / px
    return np.meshgrid(xs, ys)


def to_pixel(x, y, shape, px):
    """Scene units -> fractional (row, col)."""
    h, w = shape
    return h / 2 - y * px - 0.5, x * px + w / 2 - 0.5


def to_scene(row, col, shape, px):
    """Fractional (row, col) -> scene units."""
    h, w = shape
    return (col + 0.5 - w / 2) / px, (h / 2 - row - 0.5) / px


def disk_image(shape, px, disks, ellipses=(), ss=4):
    """BG plus disks [(cx, cy, r, contrast)] and ellipses [(cx, cy, a, b, angle_deg, contrast)]
    (semi-axes a, b) in scene units, anti-aliased by ss x ss supersampling."""
    img = np.zeros(shape)
    X0, Y0 = pixel_grid(shape, px)
    for a in range(ss):
        for b in range(ss):
            X = X0 + ((b + 0.5) / ss - 0.5) / px
            Y = Y0 - ((a + 0.5) / ss - 0.5) / px
            for cx, cy, r, c in disks:
                img += c * ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r)
            for cx, cy, ea, eb, ang, c in ellipses:
                ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                u = (X - cx) * ca + (Y - cy) * sa
                v = -(X - cx) * sa + (Y - cy) * ca
                img += c * ((u / ea) ** 2 + (v / eb) ** 2 <= 1)
    return BG + img / (ss * ss)


def blur2d(img, sigma_px):
    """Gaussian blur (std sigma_px pixels) by FFT on a copy padded by repeating the edge pixels."""
    if sigma_px <= 0:
        return img.copy()
    p = int(math.ceil(4 * sigma_px)) + 1
    big = np.pad(img, p, mode="edge")
    fy = np.fft.fftfreq(big.shape[0])[:, None]
    fx = np.fft.rfftfreq(big.shape[1])[None, :]
    tf = np.exp(-2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2))
    out = np.fft.irfft2(np.fft.rfft2(big) * tf, s=big.shape)
    return out[p:-p, p:-p]


def disk_center_blur(r, sigma, contrast=PEAK - BG):
    """Exact value of an isolated disk (radius r) blurred by sigma, at the disk center."""
    return BG + contrast * (1 - math.exp(-r * r / (2 * sigma * sigma)))


def disk_norm_log(r, sigma, contrast=PEAK - BG):
    """Scale-normalized cooling rate -sigma^2 lap(G_sigma * disk) at the center: contrast * 2z e^-z, z = r^2/2sigma^2.

    Peaks at sigma = r / sqrt(2) with height 2 * contrast / e, the same for every radius.
    """
    z = r * r / (2 * sigma * sigma)
    return contrast * 2 * z * math.exp(-z)


def disk_norm_dog(r, sigma, k=SIFT_K, contrast=PEAK - BG):
    """-(G(k sigma) - G(sigma)) * disk / (k - 1) at the center: the DoG estimate of disk_norm_log."""
    return (disk_center_blur(r, sigma, contrast) - disk_center_blur(r, k * sigma, contrast)) / (k - 1)


def dog_stack(img, sigma0_px, n_layers, k=SIFT_K):
    """SIFT DoG layers D_i = L(k^(i+1) sigma0) - L(k^i sigma0), i < n_layers.

    Returns (D, sigmas) where D has shape (n_layers, H, W) and sigmas[i] = k^i sigma0 (pixels);
    layer i is a centered difference at the geometric mid-scale sqrt(k) * sigmas[i].
    """
    sig = sigma0_px * k ** np.arange(n_layers + 1)
    L = np.array([blur2d(img, s) for s in sig])
    return L[1:] - L[:-1], sig[:-1]


def neighbors26(D, i, r, c):
    """The 3x3x3 block of D centered on (layer i, row r, col c)."""
    return D[i - 1:i + 2, r - 1:r + 2, c - 1:c + 2]


def find_extrema(D, border=2):
    """All samples strictly above or strictly below their 26 neighbors (interior layers only).

    Returns a list of (layer, row, col)."""
    n, h, w = D.shape
    core = D[1:-1, 1:-1, 1:-1]
    is_max = np.ones(core.shape, bool)
    is_min = np.ones(core.shape, bool)
    for dl in (-1, 0, 1):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dl == dr == dc == 0:
                    continue
                nb = D[1 + dl:n - 1 + dl, 1 + dr:h - 1 + dr, 1 + dc:w - 1 + dc]
                is_max &= core > nb
                is_min &= core < nb
    out = []
    for l, r, c in zip(*np.nonzero(is_max | is_min)):
        l, r, c = int(l) + 1, int(r) + 1, int(c) + 1
        if border <= r < h - border and border <= c < w - border:
            out.append((l, r, c))
    return out


def edge_ratio(D, i, r, c):
    """tr(H)^2 / det(H) of the 2x2 spatial Hessian of layer i at (r, c); inf if det <= 0."""
    d = D[i]
    dxx = d[r, c + 1] - 2 * d[r, c] + d[r, c - 1]
    dyy = d[r + 1, c] - 2 * d[r, c] + d[r - 1, c]
    dxy = (d[r + 1, c + 1] - d[r + 1, c - 1] - d[r - 1, c + 1] + d[r - 1, c - 1]) / 4
    det = dxx * dyy - dxy * dxy
    return float("inf") if det <= 0 else (dxx + dyy) ** 2 / det


def classify_extrema(D, ext, contrast=LOWE_CONTRAST, edge_r=LOWE_EDGE_R):
    """Lowe's two rejection tests. Returns one tag per extremum: 'keep', 'weak' or 'edge'."""
    lim = (edge_r + 1) ** 2 / edge_r
    tags = []
    for i, r, c in ext:
        if abs(D[i, r, c]) < contrast:
            tags.append("weak")
        elif edge_ratio(D, i, r, c) >= lim:
            tags.append("edge")
        else:
            tags.append("keep")
    return tags


def bilinear(img, rows, cols):
    """Bilinear samples of img at fractional (rows, cols), clamped to the image."""
    h, w = img.shape
    rows = np.clip(rows, 0, h - 1.000001)
    cols = np.clip(cols, 0, w - 1.000001)
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    fr, fc = rows - r0, cols - c0
    return ((1 - fr) * (1 - fc) * img[r0, c0] + (1 - fr) * fc * img[r0, c0 + 1]
            + fr * (1 - fc) * img[r0 + 1, c0] + fr * fc * img[r0 + 1, c0 + 1])


def sample_patch(img, row, col, half_px, n=16):
    """n x n bilinear samples covering the square of half-width half_px pixels around (row, col)."""
    t = (np.arange(n) + 0.5) / n * 2 - 1
    R, C = np.meshgrid(row + t * half_px, col + t * half_px, indexing="ij")
    return bilinear(img, R, C)


def patch_gradients(patch):
    """(gx, gy) of a patch by central differences, with y up (rows grow downward)."""
    gx = np.zeros_like(patch)
    gy = np.zeros_like(patch)
    gx[:, 1:-1] = (patch[:, 2:] - patch[:, :-2]) / 2
    gy[1:-1, :] = (patch[:-2, :] - patch[2:, :]) / 2
    return gx, gy


def orientation_hist(gx, gy, n_bins=8):
    """Gradient-magnitude-weighted histogram of gradient directions (bin 0 centered on +x)."""
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), 2 * np.pi)
    b = np.floor(ang / (2 * np.pi) * n_bins + 0.5).astype(int) % n_bins
    return np.bincount(b.ravel(), weights=mag.ravel(), minlength=n_bins)


def sift_descriptor(patch, cells=4, n_bins=8):
    """SIFT-style descriptor: a cells x cells grid of orientation histograms, L2-normalized,
    clipped at 0.2 and renormalized (Lowe 2004). Returns (vector, per-cell histograms)."""
    gx, gy = patch_gradients(patch)
    n = patch.shape[0] // cells
    hists = np.array([[orientation_hist(gx[a * n:(a + 1) * n, b * n:(b + 1) * n],
                                        gy[a * n:(a + 1) * n, b * n:(b + 1) * n], n_bins)
                       for b in range(cells)] for a in range(cells)])
    v = hists.ravel()
    v = v / (np.linalg.norm(v) + 1e-12)
    v = np.minimum(v, 0.2)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, hists


PATCH_K = 3.0                       # descriptor patch half-width = PATCH_K * sigma (Scene 9)


def detect_and_describe(img, px, sigma0, n_layers, k=SIFT_K, half_k=PATCH_K, n=16):
    """The whole pipeline of Scenes 7-9 on one image: DoG stack -> 26-neighbor extrema -> Lowe's two
    tests -> a patch of half-width half_k * sigma, resampled to n x n from the image blurred at sigma
    -> SIFT-style descriptor. Keypoints whose patch would leave the image are dropped.

    Returns a list of dicts: layer, row, col, sigma (scene units, layer mid-scale), half (patch
    half-width, pixels), patch (n x n), desc (vector), hists (cells x cells x bins)."""
    D, sig = dog_stack(img, sigma0 * px, n_layers, k)
    ext = find_extrema(D)
    out = []
    h, w = img.shape
    for (l, r, c), tag in zip(ext, classify_extrema(D, ext)):
        if tag != "keep":
            continue
        s_px = sig[l] * math.sqrt(k)
        half = half_k * s_px
        if min(r, c, h - 1 - r, w - 1 - c) <= half + 1:
            continue
        patch = sample_patch(blur2d(img, s_px), r, c, half, n)
        desc, hists = sift_descriptor(patch)
        out.append(dict(layer=l, row=r, col=c, sigma=s_px / px, half=half, patch=patch, desc=desc, hists=hists))
    return out


def match_ratio(desc_a, desc_b, ratio=0.8):
    """Lowe ratio-test matches between two descriptor lists: [(i, j)] with best < ratio * second best."""
    out = []
    if len(desc_b) < 2:
        return out
    B = np.array(desc_b)
    for i, d in enumerate(desc_a):
        dist = np.linalg.norm(B - d, axis=1)
        j1, j2 = np.argsort(dist)[:2]
        if dist[j1] < ratio * dist[j2]:
            out.append((i, int(j1)))
    return out
