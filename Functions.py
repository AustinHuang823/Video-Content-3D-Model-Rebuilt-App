import cv2 as cv
import numpy as np

"""
Feature tracking and the factorization method, shared by main.py (sparse Tomasi-Kanade)
and dense.py (perspective refinement and dense model).
"""

# Pyramidal Lucas-Kanade: 21x21 windows over 5 pyramid levels follow the larger motions in the
# Medusa video; a point must come back to within FB_THRESH px when tracked backwards.
LK_PARAMS = dict(winSize=(21, 21), maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 40, 0.01))
FB_THRESH = 0.5


def load_gray(files):
    imgs = [cv.imread(f) for f in files]
    return imgs, [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in imgs]


def _track(gray, seq, p):
    """Follow points p (detected in frame seq[0]) through the frames in seq."""
    pts = {seq[0]: p}
    ok = np.ones(len(p), bool)
    cur = p
    for a, b in zip(seq[:-1], seq[1:]):
        p0 = cur.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv.calcOpticalFlowPyrLK(gray[a], gray[b], p0, None, **LK_PARAMS)
        pb, st2, _ = cv.calcOpticalFlowPyrLK(gray[b], gray[a], p1, None, **LK_PARAMS)
        q = p1.reshape(-1, 2)
        h, w = gray[b].shape
        good = (st.ravel() == 1) & (st2.ravel() == 1)
        good &= np.linalg.norm(pb - p0, axis=2).ravel() < FB_THRESH
        good &= (q[:, 0] > 2) & (q[:, 1] > 2) & (q[:, 0] < w - 3) & (q[:, 1] < h - 3)
        # Points that move against the epipolar geometry of this frame pair are mismatches
        idx = np.where(ok & good)[0]
        _, inlier = cv.findFundamentalMat(cur[idx], q[idx], cv.FM_RANSAC, 1.0, 0.999)
        keep = np.zeros(len(ok), bool)
        keep[idx[inlier.ravel() == 1]] = True
        ok &= keep
        pts[b] = q
        cur = q
    return pts, ok


def track_features(gray, ref=0, first=0, last=None):
    """
    Track Shi-Tomasi corners detected in frame `ref` forwards to `last` and backwards to `first`.
    Only points followed through every frame are kept, so each column of the measurement matrix
    is one physical point. Returns the tracks, shape (frames, points, 2), for frames first..last.
    """
    last = len(gray) - 1 if last is None else last
    p = cv.goodFeaturesToTrack(gray[ref], maxCorners=30000, qualityLevel=0.001,
                               minDistance=3, blockSize=7).reshape(-1, 2)
    fwd, ok_f = _track(gray, list(range(ref, last + 1)), p)
    bwd, ok_b = _track(gray, list(range(ref, first - 1, -1)), p)
    ok = ok_f & ok_b
    pts = {**fwd, **bwd}
    return np.stack([pts[f][ok] for f in range(first, last + 1)])


def g_builder(a, b):
    """Coefficients of a^T L b in the six unknowns (l11, l12, l13, l22, l23, l33) of a symmetric L."""
    return [a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[0] * b[2] + a[2] * b[0],
            a[1] * b[1], a[1] * b[2] + a[2] * b[1], a[2] * b[2]]


def G_mat_builder(R, ref=0):
    """
    Metric constraints for scaled orthography, one pair per frame f with rows i_f, j_f of R:
        i_f^T L i_f - j_f^T L j_f = 0      (equal scale on both image axes)
        i_f^T L j_f = 0                    (perpendicular image axes)
    plus i_ref^T L i_ref = 1 to fix the overall scale. Unlike the unit-norm constraints of pure
    orthography, these allow the scale to change from frame to frame (zoom, camera moving away).
    """
    n = R.shape[0] // 2
    rows, c = [], []
    for f in range(n):
        i, j = R[f], R[n + f]
        rows.append(np.subtract(g_builder(i, i), g_builder(j, j)))
        c.append(0)
        rows.append(g_builder(i, j))
        c.append(0)
    rows.append(g_builder(R[ref], R[ref]))
    c.append(1)
    return np.array(rows), np.array(c, float)


def factorize(tracks, ref=0):
    """
    Tomasi-Kanade factorization ("Shape and Motion from Image Streams under Orthography:
    a Factorization Method") with metric constraints for scaled orthography.
    tracks: (F, P, 2). Returns motion R (2F x 3, rows i_1..i_F then j_1..j_F), shape S (3 x P),
    the registration t (2F x 1, image of the centroid) and the singular values of ~W.
    """
    # Measurement matrix W = [U; V] (2F x P), registered by subtracting each row's mean
    W = np.vstack((tracks[:, :, 0], tracks[:, :, 1]))
    t = W.mean(axis=1, keepdims=True)
    O1, s, O2 = np.linalg.svd(W - t, full_matrices=False)
    # Rank theorem for noisy measurements: keep the three largest singular values
    R_hat = O1[:, :3] * np.sqrt(s[:3])
    S_hat = np.sqrt(s[:3])[:, None] * O2[:3]
    # Solve G l = c for L = Q Q^T, then R = R_hat Q and S = Q^-1 S_hat
    G, c = G_mat_builder(R_hat, ref)
    l = np.linalg.lstsq(G, c, rcond=None)[0]
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    e_vl, e_vt = np.linalg.eigh(L)
    e_vl = np.clip(e_vl, 1e-12, None)  # enforce positive definiteness
    Q = e_vt @ np.diag(np.sqrt(e_vl))
    return R_hat @ Q, np.linalg.inv(Q) @ S_hat, t, s


def cameras_from_motion(R, t):
    """Rotation rows i, j, k, depth tz and image of the origin (x0, y0) for each frame."""
    n = R.shape[0] // 2
    I, J = R[:n], R[n:]
    tz = 2 / (np.linalg.norm(I, axis=1) + np.linalg.norm(J, axis=1))
    i = I / np.linalg.norm(I, axis=1, keepdims=True)
    j = J - i * (i * J).sum(1, keepdims=True)
    j /= np.linalg.norm(j, axis=1, keepdims=True)
    return i, j, np.cross(i, j), tz, t[:n, 0], t[n:, 0]


def project(cams, P):
    """Perspective projection of 3 x N points into normalized image coordinates, per frame."""
    i, j, k, tz, x0, y0 = cams
    den = k @ P + tz[:, None]
    return (i @ P + (x0 * tz)[:, None]) / den, (j @ P + (y0 * tz)[:, None]) / den


def perspective_factorization(x, y, ref=0, iters=50):
    """
    Upgrade the scaled-orthographic factorization to full perspective (Christy & Horaud,
    "Euclidean shape and motion from multiple perspective views by affine iterations", 1996).
    x, y: (F, P) normalized image coordinates. With eps_fp = k_f . P_p / tz_f, the perspective
    projection satisfies x (1 + eps) = scaled-orthographic projection, so we factor the corrected
    measurements, update eps from the new solution and repeat until eps stops changing.
    Each round keeps whichever of the two mirror-image affine solutions reprojects better under
    perspective, which also settles the depth reversal that affine factorization cannot.
    Returns (mean reprojection error in normalized units, cameras, shape).
    """
    eps = np.zeros_like(x)
    D = np.diag([1, 1, -1.0])
    best = None
    for _ in range(iters):
        R, S, t, _ = factorize(np.dstack((x * (1 + eps), y * (1 + eps))), ref)
        candidates = []
        for Rc, Sc in ((R, S), (R @ D, D @ S)):
            cams = cameras_from_motion(Rc, t)
            px, py = project(cams, Sc)
            err = np.sqrt((px - x) ** 2 + (py - y) ** 2).mean()
            candidates.append((err, (cams[2] @ Sc) / cams[3][:, None], cams, Sc))
        err, new_eps, cams, S = min(candidates, key=lambda c: c[0])
        done = np.abs(new_eps - eps).max() < 1e-6
        eps, best = new_eps, (err, cams, S)
        if done:
            break
    return best


def solve_focal(tracks, cx, cy, ref, focals):
    """Run the perspective factorization for each candidate focal length (px); keep the best fit."""
    results = []
    for f in focals:
        err, cams, S = perspective_factorization((tracks[:, :, 0] - cx) / f, (tracks[:, :, 1] - cy) / f, ref)
        results.append((err * f, f, cams, S))
    return min(results, key=lambda r: r[0]), results
