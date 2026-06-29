"""Load Stage-1 cached features/labels + cache-integrity checks."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import utils

SPLITS = ("train", "val", "test")


def _feat_paths(cfg, split):
    root = cfgmod.bm(cfg, "feature_root")
    ds = cfg.general.dataset_name
    ir = cfg.datasets.imbalance_ratio
    d = os.path.join(root, ds)
    return (os.path.join(d, f"{split}_{ir}_feats.npy"),
            os.path.join(d, f"{split}_{ir}_labels.npy"))


def load_extract_meta(cfg):
    """Return Stage-1 extract_meta.json dict, or None if absent."""
    root = cfgmod.bm(cfg, "feature_root")
    path = os.path.join(root, cfg.general.dataset_name, "extract_meta.json")
    return utils.load_json(path) if os.path.exists(path) else None


def load_split(cfg, split):
    """Return (feats float32 tensor [N,D], labels long tensor [N])."""
    fp, lp = _feat_paths(cfg, split)
    if not (os.path.exists(fp) and os.path.exists(lp)):
        raise FileNotFoundError(
            f"Missing Stage-1 cache for split '{split}': {fp}. "
            "Run `python -m biomedclip_ltc.extract_features` first.")
    feats = torch.from_numpy(np.load(fp)).float()
    labels = torch.from_numpy(np.load(lp)).long()
    return feats, labels


def check_cache_integrity(cfg, feats, labels, split, meta=None, l2_tol=1e-2):
    """Validate feature dim, sample count, label range, L2-normalization, and
    split provenance. Raises AssertionError on any violation; returns a report."""
    C = cfg.general.num_classes
    D = cfgmod.bm(cfg, "feature_dim")
    n = feats.shape[0]
    assert feats.dim() == 2, f"[{split}] feats must be 2-D, got {feats.shape}"
    assert feats.shape[1] == D, f"[{split}] feat dim {feats.shape[1]} != config {D}"
    assert labels.shape[0] == n, f"[{split}] labels {labels.shape[0]} != feats {n}"
    lo, hi = int(labels.min()), int(labels.max())
    assert 0 <= lo and hi < C, f"[{split}] label range [{lo},{hi}] outside [0,{C})"
    norms = feats.norm(dim=1)
    mean_norm = float(norms.mean())
    max_dev = float((norms - 1.0).abs().max())
    assert max_dev < l2_tol, (
        f"[{split}] features not L2-normalized (max |‖x‖-1| = {max_dev:.4f}); "
        "Stage 1 must L2-normalize.")
    report = {"split": split, "n": n, "feat_dim": feats.shape[1],
              "label_min": lo, "label_max": hi, "mean_l2_norm": round(mean_norm, 5),
              "max_norm_dev": round(max_dev, 5)}
    if meta is not None:
        assert meta.get("feature_dim") == D, \
            f"[{split}] extract_meta feature_dim {meta.get('feature_dim')} != {D}"
        assert meta.get("hub_id") == cfgmod.bm(cfg, "hub_id"), \
            f"[{split}] extract_meta hub_id mismatch vs config"
        exp = (meta.get("split_sizes") or {}).get(split)
        if exp is not None:
            assert exp == n, f"[{split}] size {n} != extract_meta {exp}"
        report["hub_id"] = meta.get("hub_id")
        report["prompt_version"] = meta.get("prompt_version")
        report["normalization"] = meta.get("normalization")
    return report


def get_cls_num_list(labels, num_classes):
    """Per-class training counts (for optional LT losses)."""
    counts = torch.bincount(labels, minlength=num_classes)
    return counts.tolist()


def get_logit_scale(cfg, default=100.0):
    """BiomedCLIP logit scale recorded by Stage 1 (exp(model.logit_scale)), or a
    sensible default if extract_meta.json is absent."""
    meta = load_extract_meta(cfg)
    if meta and meta.get("logit_scale"):
        return float(meta["logit_scale"])
    return float(default)


def load_text_prototypes(cfg, scheme):
    """Return text prototypes [C, D] float32 tensor for a scheme (P0/P1/P2)."""
    root = cfgmod.bm(cfg, "proto_root")
    path = os.path.join(root, f"{cfg.general.dataset_name}_{scheme}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prototypes {path}; run Stage 1.")
    return torch.from_numpy(np.load(path)).float()


class FeatureDataset(Dataset):
    """Dataset over cached (feature, label) pairs."""

    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, i):
        return self.feats[i], self.labels[i]
