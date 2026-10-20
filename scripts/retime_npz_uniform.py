"""retime_npz_uniform.py remove residual slow-motion from a retargeted motion npz so
the gait plays at a UNIFORM real-time cadence (constant speed), while keeping the whole
clip. The local gait rate is proxied by the body-relative foot sweep speed (slow-mo ->
low speed); we drop frames proportionally (the npz analog of unslowmo_video.py) so each
stride takes the same time. Positions are resampled; joint velocities/accelerations are
re-derived at the (unchanged) dt; the caller should re-run recover_root_translation.py
afterward so the root velocity is re-derived on the now-uniform gait.

  python scripts/retime_npz_uniform.py --motion <motion.npz> [--smooth 61]
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--smooth", type=int, default=61,
                help="rate-smoothing window in frames (>> one stride, to ignore within-stride swing/stance)")
ap.add_argument("--floor", type=float, default=0.3, help="min local-rate (cap how much any region is compressed)")
args = ap.parse_args()

p = Path(args.motion)
d = dict(np.load(p, allow_pickle=True))
T = int(np.asarray(d["joint_pos_isaac"]).shape[0])
dt = float(d["dt"])
fps = 1.0 / dt

# local gait rate ~ body-relative foot sweep speed, smoothed over >1 stride
fpl = np.asarray(d["foot_pos_local"])                                  # (T,4,3)
fspeed = np.linalg.norm(np.diff(fpl, axis=0, prepend=fpl[:1]), axis=2).mean(axis=1)  # (T,)
w = args.smooth
fs = np.convolve(np.pad(fspeed, (w // 2, w // 2), "edge"), np.ones(w) / w, "valid")[:T]
r = np.clip(fs / np.percentile(fs, 75), args.floor, 1.0)               # normal->1, slow-mo-><1

# cumulative-rate decimation: sample output frames evenly in accumulated gait-progress
C = np.cumsum(r)
n_out = int(C[-1])
idx = np.clip(np.searchsorted(C, np.arange(1, n_out + 1)), 0, T - 1)

# resample every per-frame array; re-derive velocity/acceleration at the SAME dt
for k, v in list(d.items()):
    a = np.asarray(v)
    if a.ndim >= 1 and a.shape[0] == T:
        d[k] = a[idx]
jp = d["joint_pos_isaac"]
d["joint_vel"] = np.gradient(jp, dt, axis=0)
d["joint_acc"] = np.gradient(d["joint_vel"], dt, axis=0)

np.savez(p, **d)
print(f"retimed {p.name}: {T} -> {n_out} frames ({T/fps:.1f}s -> {n_out/fps:.1f}s @ {fps:.0f}fps)")
print(f"  dropped {T-n_out} slow-mo frames; local rate min {r.min():.2f} (most-compressed) .. 1.00")
print(f"  NOTE: re-run recover_root_translation.py on this file to re-derive the root velocity.")
