"""unslowmo_video.py — remove variable slow-motion from a clip so the on-screen
motion plays at a uniform real-time speed (for clips that are slowed at the ends
and normal in the middle, etc.). Frame-to-frame motion energy is a proxy for the
local real-time rate (slow-mo -> low energy); we decimate frames proportionally so
the output energy is ~flat. Iterated, because a single pass under-corrects (frame
skip vs motion is sub-linear). Pure CPU (cv2). The downstream retarget then re-derives
every npz field at a consistent dt, so the learned motion comes out uniform-speed.

  python scripts/unslowmo_video.py --in <src.mp4> --out <dst.mp4> [--skip-start 3.0] [--iters 3]
"""
import argparse, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--skip-start", type=float, default=0.0, help="seconds to drop from the start (title card)")
ap.add_argument("--skip-end", type=float, default=0.0, help="seconds to drop from the end")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--smooth", type=int, default=31, help="energy smoothing window (frames; >> one gait stride)")
args = ap.parse_args()

cap = cv2.VideoCapture(args.inp); fps = cap.get(5)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.skip_start * fps))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
if args.skip_end > 0:
    frames = frames[:-int(args.skip_end * fps)]
N = len(frames)
small = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
print(f"loaded {N} frames ({N/fps:.1f}s, variable speed)")

def energy(idx):
    return np.array([0.0] + [np.abs(small[idx[i]] - small[idx[i-1]]).mean() for i in range(1, len(idx))])

def ratio(idx):  # uniformity: per-second-min / per-second-max of smoothed energy
    e = energy(idx); spf = int(fps)
    segs = [e[s:s+spf].mean() for s in range(spf, len(e)-spf, spf)]  # skip edges
    return min(segs)/max(segs) if segs else 1.0

idx = np.arange(N)
for it in range(args.iters):
    e = energy(idx)
    w = args.smooth
    es = np.convolve(np.pad(e, (w//2, w//2), "edge"), np.ones(w)/w, "valid")[:len(e)]
    r = np.clip(es / np.percentile(es, 90), 0.12, 1.0)      # normal->1, slow-mo-><1
    C = np.cumsum(r); n_out = int(C[-1])
    local = np.clip(np.searchsorted(C, np.arange(1, n_out+1)), 0, len(idx)-1)
    idx = idx[local]
    print(f"  iter {it+1}: {len(idx)} frames ({len(idx)/fps:.1f}s)  uniformity min/max={ratio(idx):.2f}")

h, wd = frames[0].shape[:2]
vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
for i in idx:
    vw.write(frames[i])
vw.release()
print(f"wrote {args.out}: {len(idx)} frames @ {fps:.0f}fps = {len(idx)/fps:.1f}s (uniform real-time)")
