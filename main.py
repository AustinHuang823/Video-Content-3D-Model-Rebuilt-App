"""
Structure from motion with the Tomasi-Kanade factorization method (EN.520.665 Machine Perception).

For each image sequence:
1. Track feature points through every frame: Shi-Tomasi corners followed by pyramidal
   Lucas-Kanade, with a forward-backward check and RANSAC on each frame pair, so every column
   of the measurement matrix is one physical point seen in all frames.
2. Register the measurement matrix and factor it with the rank-3 SVD (Tomasi & Kanade,
   "Shape and Motion from Image Streams under Orthography: a Factorization Method").
3. Upgrade the affine factors to metric motion and shape with the metric constraints, written for
   scaled orthography so the camera may zoom or change distance (it does in the Medusa video).

    python main.py            writes medusaplot.png and castleplot.png
    python main.py --show     also opens the plots

dense.py goes further on the Medusa sequence: perspective refinement and a dense 3D model.
"""
import argparse
import glob

import matplotlib
import numpy as np

import Functions

DATASETS = [
    # name, frames, output
    ('Medusa Head', sorted(glob.glob('medusajpg/medusa_out0???.jpg')), 'medusaplot.png'),
    ('Castle Sequence', sorted(glob.glob('castlejpg/castle.???.jpg')), 'castleplot.png'),
]


def reconstruct(files):
    imgs, gray = Functions.load_gray(files)
    tracks = Functions.track_features(gray, ref=0)
    R, S, t, s = Functions.factorize(tracks, ref=0)
    W = np.vstack((tracks[:, :, 0], tracks[:, :, 1]))
    err = np.abs(R @ S + t - W)
    print(f'  {len(files)} frames, {tracks.shape[1]} points tracked through all of them; '
          f'reprojection error mean {err.mean():.2f} px, median {np.median(err):.2f} px')
    # Express the shape in the first camera's frame: x right, y down, z along the viewing direction
    i = R[0] / np.linalg.norm(R[0])
    j = R[len(files)] - i * (i @ R[len(files)])
    j /= np.linalg.norm(j)
    shape = np.vstack((i, j, np.cross(i, j))) @ S
    colors = imgs[0][tracks[0, :, 1].astype(int), tracks[0, :, 0].astype(int), ::-1] / 255
    return shape, colors


def plot(name, shape, colors, out):
    import matplotlib.pyplot as plt
    x, y, z = shape[0], -shape[1], -shape[2]  # y up, z towards the camera for plotting
    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(f'Reconstructed {name} 3D Object using Tomasi-Kanade Factorization Method',
                 c='midnightblue', size=16, fontweight='bold')
    for n, (elev, azim, title) in enumerate([(20, -60, 'oblique view'), (90, -90, 'seen from the first camera')]):
        ax = fig.add_subplot(1, 2, n + 1, projection='3d', xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')
        ax.scatter(x, y, z, c=colors, marker='o', s=4, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    plt.savefig(out, dpi=100)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--show', action='store_true', help='open the plots after saving them')
    args = ap.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, files, out in DATASETS:
        print(name)
        shape, colors = reconstruct(files)
        plot(name, shape, colors, out)
        print(f'  wrote {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
