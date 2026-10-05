"""
Dense, textured 3D model of the Medusa head from the same video.

1. Track features through the frames around a reference frame and factor them (Functions.py).
2. Refine the scaled-orthographic solution to full perspective by iterated factorization
   (Christy & Horaud 1996), picking the focal length that reprojects best.
3. Dense correspondences: DIS optical flow chained from the reference frame to its neighbours.
   Every pixel is triangulated with the recovered perspective cameras over several baselines and
   kept only where it reprojects within 1.5 px in every frame used.
4. Fuse the baselines, re-derive the silhouette's rim and the gaps from the interior, smooth, and
   cut out the slab.
5. Write a textured mesh (PLY), a depth image, a still and three clips: the textured model
   turning, the bare geometry (tracked points, then the surface) and the mesh being built.

    python dense.py              results/ (needs ffmpeg on the PATH for the clip and the GIF)
"""
import os
import subprocess

import cv2 as cv
import numpy as np

import Functions
from render import Model, rotation

FILES = [f'medusajpg/medusa_out{i:04d}.jpg' for i in range(1, 51)]
REF = 20                       # reference frame (medusa_out0021.jpg): the head seen almost frontally
TRACK_HALF = 8                 # factor the frames REF-8 .. REF+8
FOCALS = [1050, 1075, 1100, 1125, 1150]
BASELINES = [8, 6, 4, 3]       # half-widths of the frame windows used for dense triangulation
MAX_REPROJ = 1.5               # px
RIM = 20                       # px of the silhouette's rim whose depth comes from the interior
OUT = 'results'


def keep_slab(h, w):
    """The slab's front face: below the cornice that runs across the top right of the reference frame."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy > 0.485 * (xx - 380) + 6) & (xx >= 10) & (xx < w - 10) & (yy >= 10) & (yy < h - 10)


def dis_flow():
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    dis.setPatchSize(12)
    dis.setPatchStride(3)
    dis.setGradientDescentIterations(25)
    dis.setVariationalRefinementIterations(10)
    dis.setVariationalRefinementAlpha(20)
    return dis


def chain_flow(gray, seq, dis):
    """Where each pixel of frame seq[0] lands in the later frames of seq, following the flow frame by frame."""
    h, w = gray[seq[0]].shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X, Y, out = xx.copy(), yy.copy(), {}
    for a, b in zip(seq[:-1], seq[1:]):
        fl = dis.calc(gray[a], gray[b], None)
        fx = cv.remap(fl[:, :, 0], X, Y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=np.nan)
        fy = cv.remap(fl[:, :, 1], X, Y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=np.nan)
        X, Y = X + fx, Y + fy
        out[b] = (X.copy(), Y.copy())
    return out


def triangulate(gray, cams, first, f, half, dis):
    """Depth of every reference pixel from the frames REF-half .. REF+half, with a validity mask."""
    i, j, k, tz, x0, y0 = cams
    h, w = gray[REF].shape
    cx, cy = w / 2, h / 2
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    lo, hi = max(first, REF - half), min(first + len(tz) - 1, REF + half)
    obs = {REF: (xx, yy)}
    obs.update(chain_flow(gray, list(range(REF, hi + 1)), dis))
    obs.update(chain_flow(gray, list(range(REF, lo - 1, -1)), dis))
    # Linear least squares per pixel: (i_f - x k_f) . P = x tz_f - Tx_f, and the same for y
    N = h * w
    ATA, ATb, seen = np.zeros((N, 3, 3)), np.zeros((N, 3)), np.zeros(N)
    for a, (X, Y) in obs.items():
        q = a - first
        xn, yn = ((X - cx) / f).ravel(), ((Y - cy) / f).ravel()
        ok = ~(np.isnan(xn) | np.isnan(yn))
        xn, yn = np.nan_to_num(xn), np.nan_to_num(yn)
        for row, c, rhs in ((i[q], xn, xn * tz[q] - x0[q] * tz[q]), (j[q], yn, yn * tz[q] - y0[q] * tz[q])):
            A = (row[None, :] - c[:, None] * k[q][None, :]) * ok[:, None]
            ATA += A[:, :, None] * A[:, None, :]
            ATb += A * (rhs * ok)[:, None]
        seen += ok
    P = np.linalg.solve(ATA + 1e-12 * np.eye(3), ATb[:, :, None])[:, :, 0]
    worst = np.zeros(N)
    for a, (X, Y) in obs.items():
        q = a - first
        den = P @ k[q] + tz[q]
        px = (P @ i[q] + x0[q] * tz[q]) / den * f + cx
        py = (P @ j[q] + y0[q] * tz[q]) / den * f + cy
        worst = np.maximum(worst, np.nan_to_num(np.hypot(px - X.ravel(), py - Y.ravel()), nan=99))
    valid = (seen == len(obs)) & (worst < MAX_REPROJ)
    # into the reference camera's coordinates
    q = REF - first
    Pc = P @ np.vstack((i[q], j[q], k[q])).T + np.array([x0[q] * tz[q], y0[q] * tz[q], tz[q]])
    print(f'  frames {lo}-{hi}: {valid.mean():.0%} of pixels triangulated, '
          f'median worst reprojection {np.median(worst[valid]):.2f} px')
    return Pc[:, 2].reshape(h, w), valid.reshape(h, w)


def fill(Z, known, region):
    """Fill the region's unknown pixels from the known ones: normalized blurs, then membrane relaxation."""
    Zk = np.where(known, Z, 0).astype(np.float32)
    M = known.astype(np.float32)
    out = np.where(known, Z, np.nan).astype(np.float32)
    for sigma in (2, 4, 8, 16, 32, 64):
        num, den = cv.GaussianBlur(Zk, (0, 0), sigma), cv.GaussianBlur(M, (0, 0), sigma)
        todo = np.isnan(out) & region & (den > 1e-3)
        out[todo] = num[todo] / den[todo]
    hole = region & ~known
    have = ~np.isnan(out)
    hole &= have
    Mh = have.astype(np.float32)
    for _ in range(200):
        smooth = cv.blur(np.where(have, out, 0), (3, 3)) / np.maximum(cv.blur(Mh, (3, 3)), 1e-6)
        out[hole] = smooth[hole]
    return np.nan_to_num(out)


def fuse(depths, f):
    """One depth map from several baselines: the widest baseline that is valid wins at each pixel."""
    h, w = depths[0][0].shape
    Z = np.full((h, w), np.nan, np.float32)
    for z, valid in depths:
        take = valid & np.isnan(Z)
        Z[take] = z[take]
    valid = ~np.isnan(Z)
    # the slab: largest connected piece of the measured area, closed, trimmed and with a smooth outline
    region = cv.morphologyEx(valid.astype(np.uint8), cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25)))
    _, lab, stats, _ = cv.connectedComponentsWithStats(region)
    region = (lab == 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])) & keep_slab(h, w)
    region = cv.erode(region.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    region = cv.GaussianBlur(region, (0, 0), 7) > 0.5
    # drop depth spikes, and the rim, where the flow is pulled along by the background
    med = cv.medianBlur(np.where(valid, Z, 0).astype(np.float32), 5)
    valid &= ~(np.abs(Z - med) > 0.01)
    dist = cv.distanceTransform(np.pad(region.astype(np.uint8), 1), cv.DIST_L2, 5)[1:-1, 1:-1]
    valid &= dist > RIM
    print(f'  model covers {region.mean():.0%} of the frame; {(region & ~valid).sum() / region.sum():.0%} of it filled from neighbours')
    Zi = fill(Z, valid, region)
    Zi = np.where(region, Zi, fill(np.where(region, Zi, 0), region, np.ones_like(region))).astype(np.float32)
    Zs = cv.GaussianBlur(cv.bilateralFilter(Zi, 9, 0.004, 4), (0, 0), 1.0)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    P = np.dstack(((xx - w / 2) / f * Zs, (yy - h / 2) / f * Zs, Zs))
    return P, region


def depth_image(P, region):
    z = P[:, :, 2]
    lo, hi = np.percentile(z[region], [2, 98])
    vis = cv.applyColorMap((255 * np.clip((hi - z) / (hi - lo), 0, 1)).astype(np.uint8), cv.COLORMAP_TURBO)
    vis[~region] = 0
    return vis


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def turntable(model, pivot, imgs, tracks, first, path, size=(720, 576), fps=25, yaw=22, pitch=-5):
    """The source video with its tracked points, cross-fading into the 3D model, which then turns."""
    teal = (196, 230, 62)
    sel = np.random.default_rng(0).choice(tracks.shape[1], min(1400, tracks.shape[1]), replace=False)
    cache = {}

    def model_frame(y, p):
        key = (round(y, 3), round(p, 3))
        if key not in cache:  # shift with the yaw keeps the slab centred while it turns
            cache[key] = model.render(rotation(y, p), pivot, size, shift=(-1.9 * y, 0))
        return cache[key]

    def video_frame(a, alpha=1.0):
        im = imgs[a].copy()
        overlay = im.copy()
        for x, y in tracks[a - first, sel]:
            cv.circle(overlay, (int(round(x * 4)), int(round(y * 4))), 7, teal, -1, lineType=cv.LINE_AA, shift=2)
        return cv.addWeighted(overlay, alpha, im, 1 - alpha, 0)

    seq = []
    for n, (y0, y1, p0, p1) in ((int(1.7 * fps), (yaw, -yaw, pitch, -pitch)), (int(1.2 * fps), (-yaw, 0, -pitch, 0))):
        seq += [('m', y0 + (y1 - y0) * ease(s / n), p0 + (p1 - p0) * ease(s / n)) for s in range(n)]
    seq += [('m', 0, 0)] * int(0.2 * fps)
    fade = int(0.5 * fps)
    seq += [('x', s / fade) for s in range(fade)]
    for n, a in enumerate(list(range(REF, REF + 9)) + list(range(REF + 7, REF - 1, -1))):
        seq += [('v', a)] * (3 if n % 2 == 0 else 2)
    seq += [('x', 1 - s / fade) for s in range(fade)]
    seq += [('m', 0, 0)] * int(0.2 * fps)
    n = int(1.2 * fps)
    seq += [('m', yaw * ease(s / n), pitch * ease(s / n)) for s in range(n)]
    frames = []
    for s in seq:
        if s[0] == 'm':
            frames.append(model_frame(s[1], s[2]))
        elif s[0] == 'v':
            frames.append(video_frame(s[1]))
        else:
            frames.append(cv.addWeighted(video_frame(REF, s[1]), s[1], model_frame(0, 0), 1 - s[1], 0))
    write_clip(frames, path, fps)
    return model_frame(yaw, pitch)


def geometry_turntable(model, pivot, points, path, size=(720, 576), fps=25, yaw=22, pitch=-5):
    """The bare reconstruction: the tracked 3D points, the dense surface growing under them, a turn."""
    def frame(y, p, surface, pts):
        return model.render(rotation(y, p), pivot, size, shift=(-1.9 * y, 0), plaster=True,
                            surface=surface, points=points, points_alpha=pts)

    seq = []
    n = int(1.2 * fps)
    seq += [(yaw - 12 * ease(s / n), pitch + 2.7 * ease(s / n), 0, 1) for s in range(n)]  # points only
    n = int(0.8 * fps)
    seq += [(yaw - 12 - 6 * s / n, pitch + 2.7 + 1.4 * s / n, ease(s / n), 1 - 0.85 * ease(s / n)) for s in range(n)]
    y0, p0 = seq[-1][0], seq[-1][1]
    n = int(1.6 * fps)
    seq += [(y0 + (-yaw - y0) * ease(s / n), p0 + (-pitch - p0) * ease(s / n), 1, 0.15 * (1 - s / n)) for s in range(n)]
    turn = len(seq)
    n = int(2.0 * fps)
    seq += [(-yaw + 2 * yaw * ease(s / n), -pitch + 2 * pitch * ease(s / n), 1, 0) for s in range(n)]
    n = int(0.6 * fps)
    seq += [(yaw, pitch, 1 - ease(s / n), ease(s / n)) for s in range(n)]  # back to the points
    # start the loop with the turn, so its first frame (a web page's poster image) shows the model
    write_clip([frame(*s) for s in seq[turn:] + seq[:turn]], path, fps)
    return frame(yaw, pitch, 1, 0)


def mesh_turntable(model, pivot, points, path, size=(720, 576), fps=25, yaw=22, pitch=-5):
    """
    The mesh built from the tracked points: the points, a wireframe spreading out from the face,
    the surface filling in under it, then a turn with the mesh lines kept on the surface.
    """
    reach = 500  # px from the middle of the head to past the slab's corners
    def frame(y, p, r_wire, r_surf, pts):
        return model.render_mesh(rotation(y, p), pivot, size, shift=(-1.9 * y, 0), grow=(r_wire, r_surf),
                                 points=points, points_alpha=pts)

    def spread(u):
        return 40 + (reach - 40) * ease(u)

    seq = []
    n = int(1.2 * fps)
    seq += [(yaw - 12 * ease(s / n), pitch + 2.7 * ease(s / n), 0, 0, 1) for s in range(n)]  # points only
    n = int(1.2 * fps)
    seq += [(yaw - 12 - 6 * s / n, pitch + 2.7 + 1.4 * s / n, spread(s / n), 0, 1 - 0.6 * ease(s / n)) for s in range(n)]
    n = int(1.0 * fps)
    seq += [((yaw - 18) * (1 - s / n), (pitch + 4.1) * (1 - s / n), reach, spread(s / n), 0.4 * (1 - ease(s / n)))
            for s in range(n)]
    y0, p0 = seq[-1][0], seq[-1][1]
    n = int(1.2 * fps)
    seq += [(y0 + (-yaw - y0) * ease(s / n), p0 + (-pitch - p0) * ease(s / n), np.inf, np.inf, 0) for s in range(n)]
    turn = len(seq)
    n = int(2.0 * fps)
    seq += [(-yaw + 2 * yaw * ease(s / n), -pitch + 2 * pitch * ease(s / n), np.inf, np.inf, 0) for s in range(n)]
    n = int(0.6 * fps)
    seq += [(yaw, pitch, reach * (1 - ease(s / n)), reach * (1 - ease(s / n)), ease(s / n)) for s in range(n)]
    # the turn first, as in geometry_turntable
    write_clip([frame(*s) for s in seq[turn:] + seq[:turn]], path, fps)
    return frame(yaw, pitch, np.inf, np.inf, 0)


def write_clip(frames, path, fps):
    """H.264 clip plus a small GIF for the README (needs ffmpeg)."""
    os.makedirs(path + '.frames', exist_ok=True)
    for idx, im in enumerate(frames):
        cv.imwrite(f'{path}.frames/f{idx:04d}.png', im)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{path}.frames/f%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart',
                    path], check=True)
    gif = path.rsplit('.', 1)[0] + '.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-vf',
                    'fps=12,scale=360:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4',
                    gif], check=True)
    for name in os.listdir(path + '.frames'):
        os.remove(os.path.join(path + '.frames', name))
    os.rmdir(path + '.frames')
    print(f'  wrote {path} ({len(frames) / fps:.1f} s) and {gif}')


def main():
    os.makedirs(OUT, exist_ok=True)
    imgs, gray = Functions.load_gray(FILES)
    h, w = gray[REF].shape
    first, last = REF - TRACK_HALF, REF + TRACK_HALF

    print('Tracking and factorization')
    tracks = Functions.track_features(gray, REF, first, last)
    print(f'  {tracks.shape[1]} points tracked through frames {first}-{last}')
    (err, f, cams, S), results = Functions.solve_focal(tracks, w / 2, h / 2, REF - first, FOCALS + [1e6])
    affine = [r[0] for r in results if r[1] == 1e6][0]
    print(f'  perspective refinement: focal length {f:.0f} px, reprojection error {err:.2f} px '
          f'(scaled orthography: {affine:.2f} px)')

    print('Dense depth')
    dis = dis_flow()
    depths = [triangulate(gray, cams, first, f, half, dis) for half in BASELINES]
    P, region = fuse(depths, f)
    cv.imwrite(f'{OUT}/medusa_depth.png', np.hstack((imgs[REF], depth_image(P, region))))

    print('Model')
    model = Model(P, region, imgs[REF], f)
    model.save_ply(f'{OUT}/medusa_model.ply')
    print(f'  wrote {OUT}/medusa_model.ply ({len(model.T)} triangles)')
    z = np.median(P[:, :, 2][region])
    pivot = np.array([(330 - w / 2) / f * z, (250 - h / 2) / f * z, z])  # the middle of the head
    turntable(model, pivot, imgs, tracks, first, f'{OUT}/medusa_3d.mp4')
    # the tracked points in the reference camera's frame, those on the slab
    q = REF - first
    i, j, k, tz, x0, y0 = cams
    pts = (np.vstack((i[q], j[q], k[q])) @ S).T + np.array([x0[q] * tz[q], y0[q] * tz[q], tz[q]])
    u, v = (pts[:, 0] / pts[:, 2] * f + w / 2).astype(int), (pts[:, 1] / pts[:, 2] * f + h / 2).astype(int)
    on = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    on[on] &= region[v[on], u[on]]
    geometry_turntable(model, pivot, pts[on], f'{OUT}/medusa_geometry.mp4')
    mesh_turntable(model, pivot, pts[on], f'{OUT}/medusa_mesh.mp4')
    still = model.render(rotation(22, -5), pivot, (1440, 1152), shift=(-1.9 * 22, 0))
    cv.imwrite(f'{OUT}/medusa_3d.png', cv.resize(still, (960, 768), interpolation=cv.INTER_AREA))
    print(f'  wrote {OUT}/medusa_3d.png')


if __name__ == '__main__':
    main()
