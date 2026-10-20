"""Lift DeepLabCut's 2D keypoints to the 3D CSV the retargeter reads, with no depth model.

CopyQuad does not estimate depth. The retargeting chain consumes a CSV with an
(x, y, z) triple per joint, and the lateral axis is pinned downstream
(`retarget.py --lateral-mode pin`), so the z column carries no information and is
written as a constant.

This was measured, not assumed: running the full pipeline on the same clip with
monocular depth (Video-Depth-Anything) and with z = 0 gives preprocessed references
whose x and z axes are bit-identical. Every difference is confined to the y axis,
which is exactly the axis `--lateral-mode pin` discards.

    python scripts/make_3d_csv.py horses_walk

Reads  data/2d_keypoints/<name>.csv        (from DLC/extract_2d_keypoints.py)
Writes data/3d_keypoints/<name>_with_depth_npz.csv

The `_with_depth_npz` suffix is kept because the retargeting scripts look for it.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_name", help="clip stem, e.g. horses_walk")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--csv-in", type=Path, default=None,
                    help="override the input 2D CSV path")
    ap.add_argument("--out", type=Path, default=None, help="override the output path")
    ap.add_argument("--z", type=float, default=0.0,
                    help="constant value for the z column (default 0.0)")
    args = ap.parse_args()

    # The filename DLC's SuperAnimal export produces.
    DLC_SUFFIX = "_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32_extracted.csv"
    csv_in = args.csv_in or (args.repo_root / "data" / "2d_keypoints" /
                             f"{args.video_name}{DLC_SUFFIX}")
    csv_out = args.out or (args.repo_root / "data" / "3d_keypoints" /
                           f"{args.video_name}_with_depth_npz.csv")
    if not csv_in.exists():
        raise SystemExit(f"2D keypoints not found: {csv_in}\n"
                         "Run DLC/extract_2d_keypoints.py first.")
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in, header=[0, 1])

    # Column layout the retargeter expects: each joint's (x, y, likelihood)
    # followed by a z, so the retargeter's reader finds the triple where it expects it.
    cols, data = [], {}
    for joint, sub in df.columns.to_list():
        cols.append((joint, sub))
        data[(joint, sub)] = df[(joint, sub)]
        if sub == "likelihood":
            cols.append((joint, "z"))
            data[(joint, "z")] = args.z

    out = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(cols))
    out.to_csv(csv_out, index=False)
    n_joints = sum(1 for _, s in cols if s == "z")
    print(f"[3D] {len(out)} frames, {n_joints} joints, z = {args.z} -> {csv_out}")


if __name__ == "__main__":
    main()
