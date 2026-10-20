# 🐾 DLC - Step 1 - 2D Keypoint Extraction

Extracts 2D animal keypoints from each source video using **DeepLabCut**'s
[`SuperAnimal-Quadruped`](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)
model, applied **zero-shot** (no per-clip training). This is the first stage of the
CopyQuad pipeline; its CSV output feeds the 3D-lift step (`scripts/make_3d_csv.py`).

## ▶️ Run

```bash
# extract keypoints for every clip in data/videos/ (or scope to one with COPYQUAD_VIDEO)
python extract_2d_keypoints.py
```

- **Input:** `data/videos/<clip>.mp4`
- **Output:** `data/2d_keypoints/<clip>_superanimal_quadruped_*.csv` (one row per frame; `x, y, likelihood` per keypoint)
- `batch_extract_2d.py` runs the extraction over a folder of clips.

When several subjects are detected (e.g. a horse **and** its rider), the pipeline
keeps the one with the highest mean keypoint confidence over the clip.

## 🦴 Keypoints used downstream

The retargeter consumes **15** of the SuperAnimal keypoints: the four paws, four
knees, four hips, and three spine points:

```
paws   : front_left_paw   front_right_paw   back_left_paw   back_right_paw
knees  : front_left_knee  front_right_knee  back_left_knee  back_right_knee
hips   : front_left_thai  front_right_thai  back_left_thai  back_right_thai
spine  : neck_base        back_middle       back_end
```

## 🛠️ Setup

`pip install -r requirements.txt` (DeepLabCut 3.x) into the project venv. See the top-level
[README](../README.md) for the full environment setup.
