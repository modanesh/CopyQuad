import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import deeplabcut
from pathlib import Path

# ==========================================
# 1. DIRECTORY SETUP & INFERENCE
# ==========================================
# Auto-detect the repo root from this file's location (DLC/extract_2d_keypoints.py
# lives one level below the root). Override with the COPYQUAD_ROOT environment
# variable so the script runs on any machine without editing the source.
REPO_ROOT = Path(os.environ.get("COPYQUAD_ROOT", Path(__file__).resolve().parents[1]))
base_data_dir = REPO_ROOT.joinpath("data")

VID_DIR = base_data_dir.joinpath("videos")
DLC_DIR = VID_DIR.joinpath("DLC_output")
OUTPUT_KEYPOINT_DIR = base_data_dir.joinpath("2d_keypoints")

# Ensure input/output directories exist
VID_DIR.mkdir(parents=True, exist_ok=True)
DLC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Process ONLY one clip if COPYQUAD_VIDEO is set (a path, or a stem found in VID_DIR);
# otherwise every .mp4 in VID_DIR. This scopes the whole script (inference, extraction,
# overlay, plotting) to the single video.
_only = os.environ.get("COPYQUAD_VIDEO")
if _only:
    _p = Path(_only).resolve()
    if not _p.exists():
        _p = (VID_DIR / (_only if _only.endswith(".mp4") else f"{_only}.mp4")).resolve()
    assert _p.exists(), f"COPYQUAD_VIDEO not found: {_p}"
    all_video_paths = [_p]
else:
    all_video_paths = [p.resolve() for p in VID_DIR.glob("*.mp4")]
    if not all_video_paths:
        raise SystemExit(
            f"No .mp4 found in {VID_DIR}.\n"
            "Put your clip at data/videos/<clip>.mp4, or scope to one clip with "
            "COPYQUAD_VIDEO=<stem-or-path>.")

for video in all_video_paths:
    print(video.stem)

for video in all_video_paths:
    assert video.exists(), video

    deeplabcut.video_inference_superanimal(
        [str(video)],
        "superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        videotype=".mp4",
        # opt-in: `DLC_VIDEO_ADAPT=1` fine-tunes the detector+pose net on this clip
        # (helps hard single-subject footage like the fast trot; slower it trains).
        video_adapt=os.environ.get("DLC_VIDEO_ADAPT", "0") == "1",
        dest_folder=str(DLC_DIR)
    )

# ==========================================
# 2. KEYPOINT EXTRACTION
# ==========================================
bodyparts = [
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
    "front_left_knee", "front_right_knee", "back_left_knee", "back_right_knee",
    "front_left_thai", "front_right_thai", "back_left_thai", "back_right_thai",
    "back_middle", "neck_base", "back_end"
]

save_csv = False

# Iterate for all video clips
for video in all_video_paths:
    # Use a wildcard (*) to catch whatever suffix DeepLabCut generates
    h5_pattern = f"{video.stem}_superanimal_quadruped*.h5"

    try:
        h5 = next(DLC_DIR.glob(h5_pattern))
    except StopIteration:
        print(f"Warning: Could not find .h5 file for {video.stem}. Skipping...")
        continue

    df = pd.read_hdf(h5)  # MultiIndex columns: (scorer, animal, bodypart, coord)

    if save_csv:
        full_csv = h5.with_suffix(".csv")
        df.to_csv(full_csv, index=False)
        print("\n✓ Wrote full CSV:", full_csv)

    df.columns = df.columns.droplevel(0)
    unique_animals_detected = df.columns.get_level_values(0).unique().tolist()

    if len(unique_animals_detected) > 1:
        print("Multiple animals detected!")

    # Iterate through all animals detected and use the average of all the frames on the front_left as the likelihood
    max_likelihood = float("-inf")
    max_likelihood_animal = "no animal"

    for i in range(len(unique_animals_detected)):
        animal_name = f'animal{i}'
        try:
            mean_likelihood = df[animal_name, 'front_left_paw', 'likelihood'].mean()
            if mean_likelihood > max_likelihood:
                max_likelihood = mean_likelihood
                max_likelihood_animal = animal_name
        except KeyError:
            continue

            # Use the most likely animal as the animal of choice
    target_animal = max_likelihood_animal
    print(target_animal)

    # Obtain the indices of the bodyparts to keep
    cols_to_keep = [
        col for col in df.columns
        if col[1] in bodyparts and col[0] == target_animal
    ]

    df_filtered = df[cols_to_keep]
    df_filtered.columns = df_filtered.columns.droplevel(0)

    new_filename = f"{h5.stem}_extracted.csv"
    out_csv = OUTPUT_KEYPOINT_DIR / new_filename
    df_filtered.to_csv(out_csv, index=False)
    print("✓ Wrote subset CSV:", out_csv)

# ==========================================
# 3. VIDEO OVERLAY GENERATION
# ==========================================
skeleton = [
    # Head/Neck/Trunk Connections
    ("back_middle", "neck_base"), ("back_middle", "back_end"),
    # Front Left Leg
    ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    # Front Right Leg
    ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    # Back Left Leg
    ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
    # Back Right Leg
    ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw"),
    # Trunk-to-Limb Connections
    ("neck_base", "front_left_thai"), ("neck_base", "front_right_thai"),
    ("back_end", "back_left_thai"), ("back_end", "back_right_thai"),
]

lik_thresh = 0.2
dot_radius = 4
line_thick = 2

for video in all_video_paths:
    video_path = video

    # Safely find the actual h5 file first using the wildcard
    h5_pattern = f"{video.stem}_superanimal_quadruped*.h5"
    try:
        h5_path = next(DLC_DIR.glob(h5_pattern))
    except StopIteration:
        print(f"Warning: Could not find .h5 file for {video.stem} in overlay step. Skipping...")
        continue

    # Dynamically build the CSV path using the actual h5 stem
    # (since that's how it was saved in Part 2)
    csv_filename = f"{h5_path.stem}_extracted.csv"
    csv_path = OUTPUT_KEYPOINT_DIR.joinpath(csv_filename)

    assert video_path.exists(), video_path
    assert csv_path.exists(), csv_path
    assert h5_path.exists(), h5_path

    # Read CSV with MultiIndex columns: (scorer, bodypart, coord)
    df = pd.read_csv(csv_path, header=[0, 1])
    parts_in_csv = set(df.columns.get_level_values(0))
    missing = [p for p in bodyparts if p not in parts_in_csv]
    if missing:
        print("WARNING: these requested bodyparts are not in CSV:", missing)

    # Build a dict of series for x,y,likelihood per part
    cols = {}
    for p in bodyparts:
        if p in parts_in_csv:
            cols[p] = {
                "x": df.xs((p, "x"), level=[0, 1], axis=1).values.squeeze(),
                "y": df.xs((p, "y"), level=[0, 1], axis=1).values.squeeze(),
                "l": df.xs((p, "likelihood"), level=[0, 1], axis=1).values.squeeze(),
            }

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = video_path.with_name(f"{video_path.stem}_overlay_subset.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(df):
            break

        # Gather points for this frame
        P = {}
        for p, c in cols.items():
            x, y, l = float(c["x"][frame_idx]), float(c["y"][frame_idx]), float(c["l"][frame_idx])
            if np.isfinite(x) and np.isfinite(y) and l >= lik_thresh:
                P[p] = (int(round(x)), int(round(y)))

        # Draw skeleton first (lines)
        for a, b in skeleton:
            if a in P and b in P:
                cv2.line(frame, P[a], P[b], (0, 255, 0), thickness=line_thick)

        # Draw joints
        for p, (xx, yy) in P.items():
            cv2.circle(frame, (xx, yy), dot_radius, (0, 0, 255), thickness=-1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Wrote overlay to:", out_path)


# ==========================================
# 4. PLOTTING FUNCTION & USAGE
# ==========================================
def plot_best_2d_frame(csv_path, likelihood_thresh=0.5, require_all=True, verbose=True):
    """
    2D-only skeleton plot. Chooses a frame where *all* joints are visible.
    If none, falls back to the frame with the maximum number of valid joints.
    """
    try:
        df = pd.read_csv(csv_path, header=[0, 1])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    joints = df.columns.get_level_values(0).unique().tolist()
    target_count = len(joints)

    def valid_joint(idx, j):
        if (j, 'x') not in df.columns or (j, 'y') not in df.columns or (j, 'likelihood') not in df.columns:
            return False
        x = df.loc[idx, (j, 'x')]
        y = df.loc[idx, (j, 'y')]
        p = df.loc[idx, (j, 'likelihood')]
        return (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(p, (int, float))
                and x != -1.0 and y != -1.0 and p >= likelihood_thresh)

    best_idx = None
    best_points = None
    best_count = -1

    # pass 1: try to find a frame where ALL joints are valid
    for idx in range(len(df)):
        frame_points = {}
        all_ok = True
        for j in joints:
            if valid_joint(idx, j):
                x = float(df.loc[idx, (j, 'x')])
                y = float(df.loc[idx, (j, 'y')])
                frame_points[j] = (x, y)
            else:
                all_ok = False
        if all_ok:
            best_idx = idx
            best_points = frame_points
            best_count = target_count
            break

    # pass 2: otherwise pick frame with MAX valid joints
    if best_points is None:
        for idx in range(len(df)):
            frame_points = {}
            for j in joints:
                if valid_joint(idx, j):
                    x = float(df.loc[idx, (j, 'x')])
                    y = float(df.loc[idx, (j, 'y')])
                    frame_points[j] = (x, y)
            if len(frame_points) > best_count:
                best_idx = idx
                best_points = frame_points
                best_count = len(frame_points)

    if best_points is None or best_count <= 0:
        print(f"No valid joints found with likelihood >= {likelihood_thresh}.")
        return

    missing = [j for j in joints if j not in best_points]
    if verbose:
        if best_count == target_count:
            print(f"Using frame {best_idx}: all {target_count} joints valid.")
        else:
            print(f"Using frame {best_idx}: {best_count}/{target_count} joints valid.")
            if missing:
                print("Missing:", ", ".join(missing))

    # Plot
    fig, ax = plt.subplots(figsize=(4, 10))
    xs = [pt[0] for pt in best_points.values()]
    ys = [pt[1] for pt in best_points.values()]

    ax.scatter(xs, ys, s=40)

    for name, (x, y) in best_points.items():
        ax.text(x + 2, y - 2, name, fontsize=7)

    for a, b in skeleton:
        if a in best_points and b in best_points:
            xa, ya = best_points[a]
            xb, yb = best_points[b]
            ax.plot([xa, xb], [ya, yb], linewidth=2)

    ax.set_title(f"2D Skeleton (Frame {best_idx})")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    pad = 20
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)

    # csv_path arrives as a str from the call site, so normalize it.
    _csv = Path(csv_path)
    out_png = _csv.parent / f"{_csv.stem}_skeleton.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[plot] wrote {out_png}")
    # Only pop a window on an interactive display; plt.show() otherwise blocks forever
    # on a headless machine, and this is the last statement in the script.
    if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        plt.show()
    plt.close()


# Usage: plot the 3D CSV for each video, if it exists. That file is written by
# step 2 (scripts/make_3d_csv.py), so on a first pass through the pipeline it is
# not there yet and both this and the cleanup below are skipped. Re-run this
# script after step 2 if you want the plot and the likelihood-stripped CSV.
KEYPOINT_3D_DIR = base_data_dir.joinpath("3d_keypoints")

for video in all_video_paths:
    csv_3d = KEYPOINT_3D_DIR.joinpath(f"{video.stem}_with_depth_npz.csv")
    if csv_3d.exists():
        plot_best_2d_frame(str(csv_3d), likelihood_thresh=0.5, require_all=True, verbose=True)
    else:
        print(f"[skip] 3D CSV not written yet (step 2 creates it): {csv_3d}")

# ==========================================
# 5. REMOVE LIKELIHOOD COLUMNS
# ==========================================
for video in all_video_paths:
    input_csv_file = KEYPOINT_3D_DIR.joinpath(f"{video.stem}_with_depth_npz.csv")
    output_csv_file = KEYPOINT_3D_DIR.joinpath(f"{video.stem}.csv")

    if not input_csv_file.exists():
        print(f"[skip] 3D CSV not written yet (step 2 creates it), no likelihood cleanup: {input_csv_file}")
        continue

    df = pd.read_csv(input_csv_file, header=[0, 1])

    columns_to_drop = [
        col for col in df.columns
        if col[1] == 'likelihood'
    ]

    df_filtered = df.drop(columns=columns_to_drop)
    df_filtered.to_csv(output_csv_file, index=False)

    print(f"✅ Success! Likelihood columns removed. Saved to: {output_csv_file}")