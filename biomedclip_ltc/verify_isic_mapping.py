"""Verify the ISIC-2019-LT label -> diagnosis mapping (no frequency guessing).

Cross-references the image names stored in MONICA's numpy/isic/dic.npy against the
OFFICIAL ISIC-2019 ground-truth one-hot CSV, so every label id is resolved from
real per-image diagnoses. Writes biomedclip_ltc/isic_label_map.json which the
feature-extraction stage requires before building ISIC text prototypes.

Ground-truth CSV: ISIC_2019_Training_GroundTruth.csv with columns
    image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
(download from https://challenge.isic-archive.com/data/#2019, same page as the
metadata CSV).

Usage:
    python -m biomedclip_ltc.verify_isic_mapping \
        --gt ~/Downloads/ISIC_2019_Training_GroundTruth.csv \
        --dic ./numpy/isic/dic.npy \
        --train ./numpy/isic/train_100.npy \
        --out ./biomedclip_ltc/isic_label_map.json
"""
import argparse
import csv
import json
import os

import numpy as np

# Official ISIC-2019 dx columns and human-readable full names.
DX_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
FULL_NAME = {
    "MEL": "melanoma",
    "NV": "melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
    "SCC": "squamous cell carcinoma",
}


def _norm_name(name):
    """Strip directory + .jpg so dic keys and GT 'image' column match."""
    base = os.path.basename(str(name))
    if base.lower().endswith(".jpg"):
        base = base[:-4]
    return base


def load_groundtruth(gt_path):
    """Return {image_id (no ext) -> dx abbrev}."""
    img2dx = {}
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        dx_cols = [c for c in DX_COLUMNS if c in cols]
        missing = [c for c in DX_COLUMNS if c not in cols]
        if missing:
            raise ValueError(
                f"GT CSV missing expected dx columns {missing}; found {cols}")
        for row in reader:
            img = _norm_name(row["image"])
            # one-hot -> argmax over the 8 dx columns
            vals = [float(row[c]) for c in dx_cols]
            dx = dx_cols[int(np.argmax(vals))]
            img2dx[img] = dx
    return img2dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.expanduser(
        "~/Downloads/ISIC_2019_Training_GroundTruth.csv"))
    ap.add_argument("--dic", default="./numpy/isic/dic.npy")
    ap.add_argument("--train", default="./numpy/isic/train_100.npy")
    ap.add_argument("--out", default="./biomedclip_ltc/isic_label_map.json")
    args = ap.parse_args()

    if not os.path.exists(args.gt):
        raise SystemExit(
            f"[ERROR] Ground-truth CSV not found at {args.gt}.\n"
            "Download ISIC_2019_Training_GroundTruth.csv from "
            "https://challenge.isic-archive.com/data/#2019 and pass --gt.")

    img2dx = load_groundtruth(args.gt)
    dic = np.load(args.dic, allow_pickle=True).item()
    train_names = set(_norm_name(n) for n in
                      np.load(args.train, allow_pickle=True))

    num_classes = max(dic.values()) + 1
    # For each label id, tally which dx its images carry (purity check).
    label_dx_counts = {l: {} for l in range(num_classes)}
    label_total = {l: 0 for l in range(num_classes)}
    label_train_total = {l: 0 for l in range(num_classes)}
    unmatched = 0
    for name, label in dic.items():
        key = _norm_name(name)
        dx = img2dx.get(key)
        if dx is None:
            unmatched += 1
            continue
        label_dx_counts[label][dx] = label_dx_counts[label].get(dx, 0) + 1
        label_total[label] += 1
        if key in train_names:
            label_train_total[label] += 1

    mapping = {}
    print(f"{'label':<6}{'abbrev':<8}{'full name':<26}"
          f"{'purity':<9}{'dic#':<8}{'train_LT#':<10}")
    print("-" * 67)
    ok = True
    for l in range(num_classes):
        counts = label_dx_counts[l]
        if not counts:
            print(f"{l:<6}{'?':<8}{'(no GT match)':<26}")
            ok = False
            continue
        dx = max(counts, key=counts.get)
        purity = counts[dx] / sum(counts.values())
        if purity < 0.999:
            ok = False
        print(f"{l:<6}{dx:<8}{FULL_NAME[dx]:<26}"
              f"{purity*100:>6.2f}%  {label_total[l]:<8}{label_train_total[l]:<10}")
        mapping[str(l)] = {
            "abbrev": dx,
            "full_name": FULL_NAME[dx],
            "purity": round(purity, 4),
            "dic_count": label_total[l],
            "train_lt_count": label_train_total[l],
            "dx_breakdown": counts,
        }
    if unmatched:
        print(f"\n[warn] {unmatched} dic images had no GT match (check name format).")
    if not ok:
        print("\n[warn] Some label has impure / missing dx — inspect dx_breakdown "
              "before trusting the map.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "source_gt": os.path.abspath(args.gt),
            "source_dic": os.path.abspath(args.dic),
            "num_classes": num_classes,
            "labels": mapping,
        }, f, indent=2)
    print(f"\nWrote verified label map -> {args.out}")


if __name__ == "__main__":
    main()
