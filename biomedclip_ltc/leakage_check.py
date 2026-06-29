"""Stage 0 - data integrity & cross-split leakage report.

Two checks, written to a JSON report:
  1. Exact image-name overlap across train/val/test (any dataset).
  2. lesion_id-level leakage (ISIC only): the same physical lesion appearing in
     more than one split. lesion_id is read from the ISIC metadata CSV and is
     used ONLY here — it never enters a prompt or the model.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/isic_100.yml
    python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/kvasir_20.yml
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import utils

SPLITS = ["train", "val", "test"]


def _id(name):
    base = os.path.basename(str(name))
    return base[:-4] if base.lower().endswith(".jpg") else base


def load_isic_lesion_ids(csv_path):
    """Return {image_id_no_ext: lesion_id} ('' when unknown)."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[_id(row["image"])] = (row.get("lesion_id", "") or "").strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metadata-csv", default=None,
                    help="override ISIC metadata CSV path (lesion check only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    if args.metadata_csv:
        cfg.biomedclip.isic_metadata_csv = args.metadata_csv
    dataset = cfg.general.dataset_name
    ir = cfg.datasets.imbalance_ratio

    # split -> set of image ids
    split_ids = {}
    for sp in SPLITS:
        names = list(np.load(cfg.datasets[sp].np_path, allow_pickle=True))
        split_ids[sp] = [_id(n) for n in names]

    report = {"dataset": dataset, "imbalance_ratio": ir,
              "split_sizes": {s: len(v) for s, v in split_ids.items()}}

    # --- check 1: exact image-name overlap ---
    sets = {s: set(v) for s, v in split_ids.items()}
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sets[a] & sets[b]
        overlaps[f"{a}&{b}"] = sorted(inter)[:20]
        report[f"name_overlap_{a}_{b}_count"] = len(inter)
    report["name_overlap_examples"] = {k: v for k, v in overlaps.items() if v}
    report["name_overlap_clean"] = all(
        report[f"name_overlap_{a}_{b}_count"] == 0
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")])

    # --- check 2: lesion_id cross-split (ISIC only) ---
    if dataset == "isic":
        meta_csv = cfgmod.bm(cfg, "isic_metadata_csv")
        if not os.path.exists(meta_csv):
            report["lesion_check"] = f"SKIPPED (metadata CSV not found at {meta_csv})"
        else:
            img2les = load_isic_lesion_ids(meta_csv)
            id2split = {}
            for sp in SPLITS:
                for i in split_ids[sp]:
                    id2split[i] = sp
            les2splits = defaultdict(set)
            les2imgs = defaultdict(list)
            n_known = n_unknown = 0
            for img, sp in id2split.items():
                les = img2les.get(img, "")
                if les:
                    les2splits[les].add(sp)
                    les2imgs[les].append((img, sp))
                    n_known += 1
                else:
                    n_unknown += 1
            crossing = {les: sorted(sps) for les, sps in les2splits.items() if len(sps) > 1}
            affected_imgs = sum(len(les2imgs[l]) for l in crossing)
            report["lesion_check"] = {
                "images_with_lesion_id": n_known,
                "images_without_lesion_id": n_unknown,
                "unique_lesions_in_splits": len(les2splits),
                "lesions_crossing_splits": len(crossing),
                "images_in_crossing_lesions": affected_imgs,
                "examples": {l: {"splits": crossing[l], "images": les2imgs[l][:6]}
                             for l in list(crossing)[:10]},
                "clean": len(crossing) == 0,
            }
    else:
        report["lesion_check"] = "N/A (no patient/lesion metadata for this dataset)"

    out = args.out or f"./biomedclip_ltc/stage0_leakage_report_{dataset}_{ir}.json"
    utils.save_json(out, report)

    # --- console summary ---
    print(f"\n===== Stage-0 leakage report: {dataset} IR={ir} =====")
    print("split sizes:", report["split_sizes"])
    print("exact name overlap clean:", report["name_overlap_clean"],
          "| counts:",
          {f"{a}&{b}": report[f"name_overlap_{a}_{b}_count"]
           for a, b in [("train", "val"), ("train", "test"), ("val", "test")]})
    lc = report["lesion_check"]
    if isinstance(lc, dict):
        print(f"lesion_id known/unknown: {lc['images_with_lesion_id']}/"
              f"{lc['images_without_lesion_id']}")
        print(f"lesions crossing splits: {lc['lesions_crossing_splits']} "
              f"(affecting {lc['images_in_crossing_lesions']} images) "
              f"-> {'CLEAN' if lc['clean'] else 'LEAKAGE FOUND'}")
        if not lc["clean"]:
            for les, info in lc["examples"].items():
                print(f"   lesion {les}: splits {info['splits']}")
    else:
        print("lesion check:", lc)
    print(f"report written -> {out}\n")


if __name__ == "__main__":
    main()
