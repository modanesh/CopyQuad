"""Uniformly resample a preprocessed reference npz (_refs/*.npz) in time.

Use case: a gait whose *aerial phases* make slow playback dynamically impossible
(gravity does not slow down). E.g. the greyhound gallop: resampled to fit the
7.5 rad/s ANYdrive budget it ended up at ~2 s/stride, i.e. 767 ms with no front
support untrackable. The fix is to speed the reference back up here and shrink
the stride amplitude in retarget.py (--collect-x/z) by the same factor, which
keeps joint velocities constant (velocity ~ amplitude x cadence) while making
the support gaps ballistic-feasible again.

Unlike scripts/retime_npz_uniform.py (VARIABLE slow-mo), this is a plain uniform
linear resample of every time-series key; fps/dt are kept, so content plays
`factor`x faster.

Usage:
  resample_ref_npz.py --input _refs/x_ref.npz --output _refs/x_ref_fast.npz --factor 2.43
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", type=float, help="speed-up factor (>1 = faster/shorter)")
    g.add_argument("--frames", type=int, help="target frame count")
    args = ap.parse_args()

    d = dict(np.load(args.input, allow_pickle=True))
    T = d["root_pos"].shape[0]
    n = args.frames if args.frames else max(2, int(round(T / args.factor)))
    t_old = np.arange(T, dtype=np.float64)
    t_new = np.linspace(0.0, T - 1, n)

    # Anti-alias: low-pass at the source rate before decimating, or frame-level noise
    # (DLC jitter) folds into the output as fake signal -> joint-velocity spikes after IK.
    from scipy.signal import savgol_filter
    factor = T / n
    aa_win = int(2 * np.ceil(factor) + 1) if factor > 1.5 else 0

    out = {}
    for k, v in d.items():
        if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == T:
            flat = v.reshape(T, -1).astype(np.float64)
            if aa_win >= 5:
                flat = savgol_filter(flat, aa_win, 3, axis=0)
            res = np.stack([np.interp(t_new, t_old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
            out[k] = res.reshape((n,) + v.shape[1:]).astype(v.dtype)
        else:
            out[k] = v
    if aa_win >= 5:
        print(f"[Resample] anti-alias savgol window {aa_win} @ source rate")
    if "root_rot" in out:  # lerp'd quaternions need renormalizing
        q = out["root_rot"].astype(np.float64)
        out["root_rot"] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(d["root_rot"].dtype)

    np.savez(args.output, **out)
    fps = float(np.asarray(d.get("fps", 30.0)).reshape(-1)[0])
    print(f"[Resample] {T} -> {n} frames (x{T / n:.2f} faster), fps kept {fps:.0f}: "
          f"{T / fps:.1f}s -> {n / fps:.1f}s  => {args.output}")


if __name__ == "__main__":
    main()
