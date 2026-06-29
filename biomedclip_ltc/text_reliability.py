"""Per-class text-prototype reliability from the TRAIN split only (Stage 5).

For training images of true class c, the mean cosine similarity to each text
prototype j gives a [C, C] matrix M. Class reliability is the own-vs-best-other
margin, squashed to (0,1):

    M[c, j]      = mean_{i: y_i = c} (x_i . t_j)
    raw[c]       = M[c, c] - max_{j != c} M[c, j]
    reliability  = sigmoid(raw / reliability_temperature)

Uses train features/labels ONLY (no val/test). Saves the reliability vector, the
raw margins, and the full M matrix (for paper heatmaps).

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.text_reliability --config biomedclip_ltc/configs/isic_100.yml \
        --text-scheme P1
    python -m biomedclip_ltc.text_reliability --config biomedclip_ltc/configs/isic_100.yml --all
"""
import argparse
import os

import torch

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import utils


def compute_class_reliability(train_feats, train_labels, prototypes,
                              num_classes, temperature=0.1):
    """Return (reliability [C], raw [C], sim_matrix [C, C]) as float tensors."""
    sims = train_feats @ prototypes.t()                 # [N, C] cosine (L2-normed)
    M = torch.zeros(num_classes, num_classes)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.any():
            M[c] = sims[mask].mean(dim=0)
        else:
            M[c] = float("nan")
    raw = torch.empty(num_classes)
    for c in range(num_classes):
        others = torch.cat([M[c, :c], M[c, c + 1:]])
        raw[c] = M[c, c] - others.max()
    reliability = torch.sigmoid(raw / temperature)
    return reliability, raw, M


def reliability_path(cfg, scheme):
    root = cfgmod.bm(cfg, "proto_root")
    return os.path.join(root, f"{cfg.general.dataset_name}_text_reliability_{scheme}.json")


def compute_and_save(cfg, scheme, temperature):
    C = cfg.general.num_classes
    tr_feats, tr_labels = datamod.load_split(cfg, "train")
    datamod.check_cache_integrity(cfg, tr_feats, tr_labels, "train",
                                  datamod.load_extract_meta(cfg))
    protos = datamod.load_text_prototypes(cfg, scheme)
    rel, raw, M = compute_class_reliability(tr_feats, tr_labels, protos, C, temperature)
    out = reliability_path(cfg, scheme)
    utils.save_json(out, {
        "dataset": cfg.general.dataset_name,
        "text_scheme": scheme,
        "reliability_temperature": temperature,
        "reliability": rel.tolist(),
        "raw_margin": raw.tolist(),
        "similarity_matrix": M.tolist(),
    })
    print(f"[reliability] {scheme}: " +
          " ".join(f"c{c}={rel[c]:.3f}" for c in range(C)))
    print(f"[reliability] saved -> {out}")
    return rel


def load_reliability(cfg, scheme):
    """Load saved class reliability [C] tensor (raises if absent)."""
    path = reliability_path(cfg, scheme)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run `python -m biomedclip_ltc.text_reliability "
            f"--config <cfg> --text-scheme {scheme}` first.")
    return torch.tensor(utils.load_json(path)["reliability"], dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--all", action="store_true", help="compute for P0,P1,P2")
    ap.add_argument("--reliability-temperature", type=float, default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    temp = (args.reliability_temperature
            if args.reliability_temperature is not None
            else float(cfgmod.bm(cfg, "reliability_temperature")))
    schemes = ["P0", "P1", "P2"] if args.all else [args.text_scheme
                                                   or cfgmod.bm(cfg, "text_scheme")]
    for s in schemes:
        compute_and_save(cfg, s, temp)


if __name__ == "__main__":
    main()
