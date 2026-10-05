import cv2 as cv
import numpy as np

"""
A small software renderer for the dense model (numpy only): a textured triangle mesh on the
reference frame's pixel grid, rasterized with a z-buffer and seen through a perspective camera.
"""


def rotation(yaw, pitch, roll=0):
    """Rotation by yaw (about y), then pitch (about x), then roll (about z), in degrees."""
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def bilinear(tex, u, v):
    H, W = tex.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    t = tex.astype(np.float32)
    return (t[y0, x0] * (1 - fx) * (1 - fy) + t[y0, x0 + 1] * fx * (1 - fy)
            + t[y0 + 1, x0] * (1 - fx) * fy + t[y0 + 1, x0 + 1] * fx * fy)


def grid_mesh(P, valid, step=1, max_edge=0.012):
    """Two triangles per grid cell whose corners are all valid; drop cells that span a depth jump."""
    H, W, _ = P.shape
    ys, xs = np.arange(0, H, step), np.arange(0, W, step)
    V = P[np.ix_(ys, xs)].reshape(-1, 3)
    M = valid[np.ix_(ys, xs)]
    h, w = M.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.c_[a, c, b], np.c_[b, c, d]])
    ok = M.ravel()[tris].all(1)
    edges = [np.linalg.norm(V[tris[:, p]] - V[tris[:, q]], axis=1) for p, q in ((0, 1), (1, 2), (0, 2))]
    ok &= np.max(edges, axis=0) < max_edge
    uv = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)
    return V, tris[ok], uv


def vertex_normals(V, tris):
    n = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, tris[:, k], n)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def rasterize(px, py, pz, T, W, H, max_box=12):
    """z-buffered rasterization of many small triangles; returns the triangle and barycentrics per pixel."""
    x0, x1, x2 = px[T[:, 0]], px[T[:, 1]], px[T[:, 2]]
    y0, y1, y2 = py[T[:, 0]], py[T[:, 1]], py[T[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    minx = np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(int)
    maxx = np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(int)
    miny = np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(int)
    maxy = np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(int)
    keep = (np.abs(area) > 1e-9) & (maxx - minx <= max_box) & (maxy - miny <= max_box)
    keep &= (maxx >= 0) & (maxy >= 0) & (minx < W) & (miny < H) & (pz[T].min(1) > 0)
    tid = np.where(keep)[0]
    x0, x1, x2, y0, y1, y2, area, minx, miny, maxx, maxy = [
        v[keep] for v in (x0, x1, x2, y0, y1, y2, area, minx, miny, maxx, maxy)]
    zt = pz[T[tid]]
    zbuf = np.full(H * W, np.inf)
    candidates = []
    if len(tid):
        for oy in range(int((maxy - miny).max()) + 1):
            for ox in range(int((maxx - minx).max()) + 1):
                sx, sy = minx + ox, miny + oy
                cx, cy = sx + 0.5, sy + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area
                w2 = 1 - w0 - w1
                inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (sx >= 0) & (sy >= 0) & (sx < W) & (sy < H)
                ii = np.where(inside)[0]
                if len(ii) == 0:
                    continue
                bary = np.c_[w0[ii], w1[ii], w2[ii]]
                z = (zt[ii] * bary).sum(1)
                pix = sy[ii] * W + sx[ii]
                np.minimum.at(zbuf, pix, z)
                candidates.append((pix, z, tid[ii], bary))
    tri = np.full(H * W, -1)
    bar = np.zeros((H * W, 3))
    for pix, z, t, bary in candidates:
        win = z <= zbuf[pix]
        tri[pix[win]] = t[win]
        bar[pix[win]] = bary[win]
    return tri, bar


class Model:
    """Textured mesh of a depth map on the reference frame, in that camera's coordinates."""

    def __init__(self, P, valid, texture, f, step=1, max_edge=0.012):
        self.texture, self.f = texture, f
        self.h, self.w = texture.shape[:2]
        self.V, self.T, self.uv = grid_mesh(P, valid, step, max_edge)
        self.N = vertex_normals(self.V, self.T)
        # cavity: how far each vertex sits behind its surroundings, to darken creases in the plaster look
        Z = P[:, :, 2].astype(np.float32)
        cav = sum((Z - cv.GaussianBlur(Z, (0, 0), s)) / s for s in (3, 8, 20))
        self.cavity = cv.GaussianBlur(cav, (0, 0), 1)[::step, ::step].ravel()

    def _surface(self, R, pivot, size, shift, ss):
        """Rasterize the model turned by R about `pivot`: what each covered pixel sees of the mesh."""
        W, H = size[0] * ss, size[1] * ss
        V = (self.V - pivot) @ R.T + pivot
        N = self.N @ R.T
        f = self.f * W / self.w
        cx, cy = W / 2 + shift[0] * ss, H / 2 + shift[1] * ss
        with np.errstate(divide='ignore', invalid='ignore'):
            px, py = f * V[:, 0] / V[:, 2] + cx, f * V[:, 1] / V[:, 2] + cy
        tri, bary = rasterize(px, py, V[:, 2], self.T, W, H)
        hit = tri >= 0
        t, b = self.T[tri[hit]], bary[hit]
        n = (N[t] * b[:, :, None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        pos = (V[t] * b[:, :, None]).sum(1)
        n[(n * pos).sum(1) > 0] *= -1  # shade the side facing the camera
        uv = (self.uv[t] * b[:, :, None]).sum(1)
        cavity = (self.cavity[t] * b).sum(1)
        return dict(W=W, H=H, f=f, cx=cx, cy=cy, hit=hit, n=n, pos=pos, uv=uv, cavity=cavity)

    @staticmethod
    def _lambert(n, light):
        L = np.array(light, float) / np.linalg.norm(light)
        return np.clip(n @ L, 0, 1)

    @staticmethod
    def _points(img, s, zbuf, points, alpha, color, R, pivot, ss):
        """Draw 3D points as dots where the surface (zbuf) does not hide them."""
        Q = (points - pivot) @ R.T + pivot
        qx, qy = s['f'] * Q[:, 0] / Q[:, 2] + s['cx'], s['f'] * Q[:, 1] / Q[:, 2] + s['cy']
        ix, iy = np.clip(qx.astype(int), 0, s['W'] - 1), np.clip(qy.astype(int), 0, s['H'] - 1)
        seen = Q[:, 2] <= zbuf.reshape(s['H'], s['W'])[iy, ix] + 0.004
        overlay = img.copy()
        for x, y in zip(qx[seen], qy[seen]):
            cv.circle(overlay, (int(round(x * 4)), int(round(y * 4))), int(2.6 * ss * 4), color, -1,
                      lineType=cv.LINE_AA, shift=2)
        return img * (1 - alpha) + overlay * alpha

    def render(self, R, pivot, size, shift=(0, 0), ss=2, bg=(25, 17, 12), plaster=False,
               surface=1.0, points=None, points_alpha=0.0, point_color=(196, 230, 62)):
        """
        Turn the model by R about `pivot` and render it from the reference camera (BGR uint8).
        plaster=True shades the bare geometry instead of the video texture. `points` (N x 3, same
        frame as the model) are drawn on top where the surface does not hide them; `surface` and
        `points_alpha` fade the two layers.
        """
        s = self._surface(R, pivot, size, shift, ss)
        hit, n = s['hit'], s['n']
        if plaster:  # key and fill light on an off-white material, creases darkened
            shade = 0.22 + 0.8 * self._lambert(n, (-0.6, -0.5, -0.6)) + 0.25 * self._lambert(n, (0.7, -0.2, -0.5))
            shade *= np.clip(1 + 300 * s['cavity'], 0.35, 1.15)
            color = np.array([204, 214, 219], np.float32)[None, :] * shade[:, None]
        else:
            shade = 0.7 + 0.45 * self._lambert(n, (-0.5, -0.55, -0.67))
            color = bilinear(self.texture, s['uv'][:, 0], s['uv'][:, 1]) * shade[:, None]
        out = np.empty((s['H'] * s['W'], 3), np.float32)
        out[:] = bg
        out[hit] = np.array(bg, np.float32) * (1 - surface) + np.clip(color, 0, 255) * surface
        out = out.reshape(s['H'], s['W'], 3)
        if points is not None and points_alpha > 0:
            zbuf = np.full(s['H'] * s['W'], np.inf)
            if surface >= 0.05:
                zbuf[hit] = s['pos'][:, 2]
            out = self._points(out, s, zbuf, points, points_alpha, point_color, R, pivot, ss)
        out = cv.resize(out, size, interpolation=cv.INTER_AREA)
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_mesh(self, R, pivot, size, shift=(0, 0), ss=2, bg=(25, 17, 12), grow=(np.inf, np.inf),
                    centre=(330, 250), step=12, wire=0.55, points=None, points_alpha=0.0,
                    point_color=(196, 230, 62)):
        """
        The mesh being built: a wireframe (a triangle grid every `step` px of the reference frame,
        drawn on the surface itself, so hidden lines stay hidden) and the plaster surface under it.
        grow = (r_wire, r_surf): how far from `centre` (reference-frame px) the wireframe and the
        surface have spread; `wire` is the opacity of the lines once the surface is in.
        """
        s = self._surface(R, pivot, size, shift, ss)
        hit, n, uv = s['hit'], s['n'], s['uv']
        bg = np.array(bg, np.float32)
        teal, dark_teal = np.array(point_color, np.float32), np.array((112, 132, 36), np.float32)

        def lines(c, width):
            r = np.abs((c + step / 2) % step - step / 2)
            return np.clip(1 - (r - width / 2) / 0.6, 0, 1)

        grid = np.maximum.reduce([lines(uv[:, 0], 0.9), lines(uv[:, 1], 0.9), lines(uv[:, 0] - uv[:, 1], 1.27)])
        dist = np.hypot(uv[:, 0] - centre[0], uv[:, 1] - centre[1])
        a_wire = np.clip((grow[0] - dist) / 40, 0, 1)[:, None]
        a_surf = np.clip((grow[1] - dist) / 40, 0, 1)[:, None]
        glow = np.exp(-((dist - grow[0]) / 18) ** 2)[:, None]  # the wireframe's growing edge
        # raking light from the upper left brings out the relief
        lit = 0.18 + 0.95 * self._lambert(n, (-0.85, -0.45, -0.28)) + 0.18 * self._lambert(n, (0.6, -0.1, -0.8))
        lit = np.clip(lit * np.clip(1 + 300 * s['cavity'], 0.35, 1.15), 0, 1.2)[:, None]
        base = bg * 1.25 * (1 - a_surf) + np.array([204, 214, 219], np.float32) * lit * a_surf
        line_color = teal * np.clip(0.3 + 0.8 * lit, 0.3, 1.1) * (1 - a_surf) + dark_teal * np.clip(lit, 0.5, 1.1) * a_surf
        a_line = grid[:, None] * a_wire * (1 - a_surf * (1 - wire))
        color = base * (1 - a_line) + line_color * a_line + teal * 0.54 * glow * grid[:, None]
        shown = np.maximum(a_wire, a_surf)
        out = np.empty((s['H'] * s['W'], 3), np.float32)
        out[:] = bg
        out[hit] = bg * (1 - shown) + np.clip(color, 0, 255) * shown
        out = out.reshape(s['H'], s['W'], 3)
        if points is not None and points_alpha > 0:
            zbuf = np.full(s['H'] * s['W'], np.inf)
            zbuf[hit] = np.where(shown[:, 0] > 0.5, s['pos'][:, 2], np.inf)
            out = self._points(out, s, zbuf, points, points_alpha, tuple(int(c) for c in point_color), R, pivot, ss)
        out = cv.resize(out, size, interpolation=cv.INTER_AREA)
        return np.clip(out, 0, 255).astype(np.uint8)

    def save_ply(self, path):
        """The mesh with per-vertex colours, as binary PLY (opens in MeshLab, Blender, ...)."""
        used = np.unique(self.T)
        remap = np.full(len(self.V), -1)
        remap[used] = np.arange(len(used))
        uv = self.uv[used]
        col = self.texture[uv[:, 1].astype(int), uv[:, 0].astype(int), ::-1]
        vert = np.empty(len(used), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        # y up and z towards the viewer, as most mesh viewers expect
        vert['x'], vert['y'], vert['z'] = self.V[used, 0], -self.V[used, 1], -self.V[used, 2]
        vert['red'], vert['green'], vert['blue'] = col[:, 0], col[:, 1], col[:, 2]
        faces = np.empty(len(self.T), dtype=[('n', 'u1'), ('i', '<i4', (3,))])
        faces['n'] = 3
        faces['i'] = remap[self.T][:, ::-1]  # keep the winding counter-clockwise after flipping y and z
        with open(path, 'wb') as fh:
            fh.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vert)}\n'
                      'property float x\nproperty float y\nproperty float z\n'
                      'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                      f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
            fh.write(vert.tobytes())
            fh.write(faces.tobytes())
