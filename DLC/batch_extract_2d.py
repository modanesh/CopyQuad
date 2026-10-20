"""batch_extract_2d.py run DLC SuperAnimal-Quadruped over many clips.

This is the batch version of extract_2d_keypoints.py. Where that script globs
data/videos/*.mp4 (top level only) and is wired for the single dog clip, this
driver iterates data/videos/clips_index.csv, so it picks up clips that live in
dataset subfolders (data/videos/dlc_horse/, data/videos/tigdog/, ...) and runs
the SAME model + the SAME single-animal selection as the original script.

For every clip it writes:
  * data/2d_keypoints/<stem>_superanimal_quadruped_..._hrnet_w32_extracted.csv
  * <overlay-dir>/<stem>_overlay_labeled.mp4   (side-colored, for verification)

Run in the project venv.

Verify-first subset (recommended before committing to all 131 clips):
    python3 DLC/batch_extract_2d.py --per-dataset 5

Then eyeball the labeled overlays in data/videos/_verify_overlays/ to confirm
the HORSE (not the person leading it) is the selected animal. Once happy:
    python3 DLC/batch_extract_2d.py            # all clips, skips finished ones
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import deeplabcut

REPO_ROOT = Path(os.environ.get("COPYQUAD_ROOT", Path(__file__).resolve().parents[1]))
VID_DIR = REPO_ROOT / "data" / "videos"
DLC_DIR = VID_DIR / "DLC_output"
OUTPUT_KEYPOINT_DIR = REPO_ROOT / "data" / "2d_keypoints"
CLIPS_INDEX = VID_DIR / "clips_index.csv"

H5_SUFFIX = "_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32.h5"
CSV_SUFFIX = "_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32_extracted.csv"

BODYPARTS = [
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
    "front_left_knee", "front_right_knee", "back_left_knee", "back_right_knee",
    "front_left_thai", "front_right_thai", "back_left_thai", "back_right_thai",
    "back_middle", "neck_base", "back_end",
]

# ---- labeled-overlay drawing (mirrors scripts/overlay_labeled_keypoints.py) ----
SKELETON = [
    ("back_middle", "neck_base"), ("back_middle", "back_end"),
    ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
    ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw"),
    ("neck_base", "front_left_thai"), ("neck_base", "front_right_thai"),
    ("back_end", "back_left_thai"), ("back_end", "back_right_thai"),
]
COLOR_LEFT, COLOR_RIGHT, COLOR_TRUNK = (255, 80, 0), (0, 0, 255), (0, 255, 255)
SHORT = {
    "front_left_paw": "FL_paw", "front_left_knee": "FL_knee", "front_left_thai": "FL_thigh",
    "front_right_paw": "FR_paw", "front_right_knee": "FR_knee", "front_right_thai": "FR_thigh",
    "back_left_paw": "BL_paw", "back_left_knee": "BL_knee", "back_left_thai": "BL_thigh",
    "back_right_paw": "BR_paw", "back_right_knee": "BR_knee", "back_right_thai": "BR_thigh",
    "neck_base": "neck", "back_middle": "mid", "back_end": "tail",
}


def joint_color(name: str):
    return COLOR_LEFT if "left" in name else COLOR_RIGHT if "right" in name else COLOR_TRUNK


def overlay_style(w: int, h: int) -> dict:
    """Scale every glyph (font, circle, line, outline) to the frame size so labels
    stay legible on both tiny clips (tigdog 400x224) and huge ones (dlc_horse
    2208x1242). s == 1.0 at 1920x1080, which reproduces the originally tuned sizes
    that were approved on the dog clip."""
    s = (w * w + h * h) ** 0.5 / 2200.0
    return {
        "s": s,
        "font": float(np.clip(0.6 * s, 0.30, 1.4)),
        "txt_fill": max(1, round(s)),
        "txt_outline": max(1, round(s)) + max(1, round(2 * s)),
        "radius": max(2, round(6 * s)),
        "line": max(1, round(2 * s)),
        "off": max(3, round(8 * s)),
        "leg_font": float(np.clip(0.8 * s, 0.4, 1.6)),
        "leg_fill": max(1, round(2 * s)),
        "leg_outline": max(1, round(2 * s)) + max(1, round(2 * s)),
    }


def resolve_clip_path(out_file: str) -> Path:
    """Map a clips_index out_file (e.g. 'video_data/dlc_horse/dlc_horse_001.mp4')
    to its on-disk location under data/videos/<subfolder>/<file>."""
    parts = Path(out_file).parts
    return VID_DIR / parts[-2] / parts[-1] if len(parts) >= 2 else VID_DIR / parts[-1]


def load_clips(args) -> list[tuple[str, Path]]:
    """Return [(stem, video_path)] after applying --datasets / --per-dataset /
    --clips / --limit filters. Single source of truth = clips_index.csv."""
    df = pd.read_csv(CLIPS_INDEX)
    rows = []
    for _, r in df.iterrows():
        vp = resolve_clip_path(str(r["out_file"]))
        rows.append((vp.stem, vp.parent.name, vp))

    if args.clips:
        wanted = set(args.clips)
        rows = [t for t in rows if t[0] in wanted]
    if args.datasets:
        keep = {d.lower() for d in args.datasets}
        rows = [t for t in rows if t[1].lower() in keep]
    if args.per_dataset:
        seen: dict[str, int] = {}
        kept = []
        for t in rows:
            n = seen.get(t[1], 0)
            if n < args.per_dataset:
                kept.append(t)
                seen[t[1]] = n + 1
        rows = kept
    if args.limit:
        rows = rows[: args.limit]
    return [(stem, vp) for stem, _, vp in rows]


def select_animal(df: pd.DataFrame) -> tuple[str, float, int]:
    """Same selection as extract_2d_keypoints.py: keep the animal id whose
    front_left_paw has the highest mean likelihood over the clip."""
    animals = df.columns.get_level_values(0).unique().tolist()
    best, best_lh = "animal0", float("-inf")
    for a in animals:
        try:
            lh = float(df[a, "front_left_paw", "likelihood"].mean())
        except KeyError:
            continue
        if lh > best_lh:
            best_lh, best = lh, a
    return best, best_lh, len(animals)


def extract_one(stem: str, h5: Path) -> Path:
    df = pd.read_hdf(h5)
    df.columns = df.columns.droplevel(0)  # drop scorer -> (animal, bodypart, coord)
    target, lh, n_animals = select_animal(df)
    print(f"    {n_animals} animal(s) detected; chose '{target}' "
          f"(mean front_left_paw likelihood {lh:.3f})")

    cols = [c for c in df.columns if c[0] == target and c[1] in BODYPARTS]
    sub = df[cols].copy()
    sub.columns = sub.columns.droplevel(0)  # -> (bodypart, coord)
    out_csv = OUTPUT_KEYPOINT_DIR / f"{stem}{CSV_SUFFIX}"
    sub.to_csv(out_csv, index=False)
    return out_csv


def write_labeled_overlay(stem: str, video_path: Path, csv_path: Path,
                          overlay_dir: Path, lik_thresh: float):
    df = pd.read_csv(csv_path, header=[0, 1])
    joints = df.columns.get_level_values(0).unique().tolist()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = overlay_dir / f"{stem}_overlay_labeled.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    st = overlay_style(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    leg_dy = max(14, round(st["leg_font"] * 34))
    leg_x = max(8, round(10 * st["s"]))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(df):
            break
        pts = {}
        for j in joints:
            x, y, lh = (float(df.loc[idx, (j, "x")]), float(df.loc[idx, (j, "y")]),
                        float(df.loc[idx, (j, "likelihood")]))
            if np.isfinite(x) and np.isfinite(y) and lh >= lik_thresh:
                pts[j] = (int(round(x)), int(round(y)))
        for a, b in SKELETON:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), st["line"])
        for j, (x, y) in pts.items():
            c = joint_color(j)
            cv2.circle(frame, (x, y), st["radius"], c, -1)
            label = SHORT.get(j, j)
            org = (x + st["off"], y - st["off"] // 2)
            cv2.putText(frame, label, org, font, st["font"], (255, 255, 255), st["txt_outline"], cv2.LINE_AA)
            cv2.putText(frame, label, org, font, st["font"], c, st["txt_fill"], cv2.LINE_AA)
        for i, (txt, c) in enumerate([("LEFT = blue", COLOR_LEFT), ("RIGHT = red", COLOR_RIGHT), ("trunk = yellow", COLOR_TRUNK)]):
            org = (leg_x, leg_dy + leg_dy * i)
            cv2.putText(frame, txt, org, font, st["leg_font"], (255, 255, 255), st["leg_outline"], cv2.LINE_AA)
            cv2.putText(frame, txt, org, font, st["leg_font"], c, st["leg_fill"], cv2.LINE_AA)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return out_mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="filter by subfolder name (e.g. dlc_horse tigdog)")
    ap.add_argument("--per-dataset", type=int, default=None,
                    help="process only the first N clips of each dataset (verify-first subset)")
    ap.add_argument("--clips", nargs="*", default=None, help="explicit stems, e.g. dlc_horse_001")
    ap.add_argument("--limit", type=int, default=None, help="global cap on number of clips")
    ap.add_argument("--likelihood_thresh", type=float, default=0.2, help="overlay draw threshold")
    ap.add_argument("--overlay-dir", type=Path, default=VID_DIR / "_verify_overlays")
    ap.add_argument("--no-overlay", action="store_true", help="skip labeled-overlay generation")
    ap.add_argument("--force", action="store_true", help="re-run clips whose extracted CSV already exists")
    args = ap.parse_args()

    DLC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

    clips = load_clips(args)
    if not clips:
        print("No clips matched the filters."); return
    print(f"[Batch2D] {len(clips)} clip(s) selected. Outputs -> {OUTPUT_KEYPOINT_DIR}\n")

    done = skipped = failed = 0
    for i, (stem, vp) in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {stem}  ({vp})")
        if not vp.exists():
            print(f"    !! video missing, skipping"); failed += 1; continue
        out_csv = OUTPUT_KEYPOINT_DIR / f"{stem}{CSV_SUFFIX}"
        if out_csv.exists() and not args.force:
            print("    extracted CSV exists, skipping (use --force to redo)"); skipped += 1; continue
        try:
            h5 = next(DLC_DIR.glob(f"{stem}{H5_SUFFIX}"), None)
            if h5 is None or args.force:
                deeplabcut.video_inference_superanimal(
                    [str(vp)], "superanimal_quadruped",
                    model_name="hrnet_w32", detector_name="fasterrcnn_resnet50_fpn_v2",
                    videotype=".mp4", video_adapt=False, dest_folder=str(DLC_DIR),
                )
                h5 = next(DLC_DIR.glob(f"{stem}{H5_SUFFIX}"))
            csv_path = extract_one(stem, h5)
            print(f"    wrote {csv_path.name}")
            if not args.no_overlay:
                ov = write_labeled_overlay(stem, vp, csv_path, args.overlay_dir, args.likelihood_thresh)
                print(f"    wrote overlay {ov}")
            done += 1
        except Exception as e:
            print(f"    !! FAILED: {e}"); failed += 1

    print(f"\n[Batch2D] done={done} skipped={skipped} failed={failed}")
    if not args.no_overlay and done:
        print(f"[Batch2D] VERIFY the labeled overlays in {args.overlay_dir} "
              f"-> confirm the horse (not the person) is tracked before batching the rest.")


if __name__ == "__main__":
    main()
