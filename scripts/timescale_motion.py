"""Time-scale a retargeted motion npz by a factor s (s<1 = slower).

Slowing playback by s keeps the SAME pose sequence but stretches time, so every
joint velocity scales by s and the feasibility-screen peak-velocity drops by s.
This is the causal test for the screen: slow the gallop until it passes, retrain,
and see whether it then tracks. Poses (joint_pos, foot_pos_local, root_pos,
root_rot) are unchanged; only frame_duration/fps and the velocity-derived fields
scale. Pure NumPy.

Usage:
  timescale_motion.py --input .../greyhound_gallop.npz --output .../greyhound_gallop_0p7x.npz --factor 0.7
"""
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--factor", type=float, required=True, help="s<1 slows down (velocities *= s)")
    a = ap.parse_args()
    s = a.factor
    d = dict(np.load(a.input, allow_pickle=True))

    dt_old = float(np.asarray(d["frame_duration"]).ravel()[0])
    d["frame_duration"] = np.float32(dt_old / s)              # each frame lasts longer -> slower
    if "fps" in d:
        d["fps"] = np.float32(float(np.asarray(d["fps"]).ravel()[0]) * s)
    # velocity-derived fields scale with s (v = dq/dt, and dt -> dt/s)
    if "joint_vel" in d:  d["joint_vel"] = (np.asarray(d["joint_vel"], np.float64) * s).astype(np.float32)
    if "joint_acc" in d:  d["joint_acc"] = (np.asarray(d["joint_acc"], np.float64) * s * s).astype(np.float32)
    if "root_lin_vel" in d:  d["root_lin_vel"] = (np.asarray(d["root_lin_vel"], np.float64) * s).astype(np.float32)
    # poses (joint_pos_isaac, joint_pos, foot_pos_local, root_pos, root_rot) are UNCHANGED:
    # the same displacement per frame over a longer dt -> identical trajectory, slower.

    np.savez(a.output, **d)
    q = np.asarray(d["joint_pos_isaac"], np.float64)
    qd = np.abs(np.gradient(q, axis=0) * (1.0 / float(d["frame_duration"])))
    print(f"[TimeScale] s={s}: {len(q)} frames, dt {dt_old:.4f}->{float(d['frame_duration']):.4f}s, "
          f"duration {len(q)*dt_old:.2f}->{len(q)/ (1.0/float(d['frame_duration'])):.2f}s")
    print(f"[TimeScale] peak |joint vel| now {qd.max():.2f} rad/s (screen limit 7.5) -> {a.output}")


if __name__ == "__main__":
    main()
