"""Per-class text-prototype reliability from the TRAIN split only (Stage 5).

For every training image, compute the calibrated text margin of the true class
against the best competing class:

    margin_i = z_text[i, y_i] / Tt - max_{c != y_i} z_text[i, c] / Tt

Class reliability is the per-class mean margin squashed to [0, 1]:

    R_c = sigmoid(mean_{i:y_i=c}(margin_i) / reliability_temperature)

Uses train features/labels ONLY (no val/test). Saves a compatibility JSON plus
`class_reliability_<scheme>.pt` and `reliability_metadata_<scheme>.json`.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.text_reliability --config biomedclip_ltc/configs/isic_100.yml \
        --text-scheme P1
    python -m biomedclip_ltc.text_reliability --config biomedclip_ltc/configs/isic_100.yml --all
"""
import argparse
import os

import torch
import torch.nn.functional as F

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import utils

RELIABILITY_DEFINITION = "sample_text_margin_v2"


def estimate_class_text_reliability(train_text_logits, train_labels, num_classes,
                                    text_temperature,
                                    reliability_temperature=1.0):
    """Return train-only (reliability, class_margin, class_count, sample_margins)."""
    if text_temperature <= 0:
        raise ValueError(f"text_temperature must be positive, got {text_temperature}")
    if reliability_temperature <= 0:
        raise ValueError("reliability_temperature must be positive, "
                         f"got {reliability_temperature}")

    scaled = train_text_logits / float(text_temperature)
    true_logits = scaled.gather(1, train_labels[:, None]).squeeze(1)
    mask = F.one_hot(train_labels, num_classes=num_classes).bool()
    other_logits = scaled.masked_fill(mask, float("-inf"))
    max_other = other_logits.max(dim=1).values
    margins = true_logits - max_other

    class_margin = torch.zeros(num_classes, device=scaled.device)
    class_count = torch.zeros(num_classes, device=scaled.device)
    class_margin.scatter_add_(0, train_labels, margins)
    class_count.scatter_add_(0, train_labels, torch.ones_like(margins))
    class_margin = class_margin / class_count.clamp_min(1.0)

    reliability = torch.sigmoid(class_margin / float(reliability_temperature))
    return reliability, class_margin, class_count, margins


def mean_text_logits_by_true_class(train_text_logits, train_labels, num_classes):
    """Class x class mean text-logit matrix, kept for heatmaps/diagnostics."""
    M = torch.zeros(num_classes, train_text_logits.shape[1],
                    device=train_text_logits.device)
    counts = torch.zeros(num_classes, device=train_text_logits.device)
    M.scatter_add_(0, train_labels[:, None].expand_as(train_text_logits),
                   train_text_logits)
    counts.scatter_add_(0, train_labels, torch.ones_like(train_labels, dtype=M.dtype))
    return M / counts[:, None].clamp_min(1.0)


def reliability_path(cfg, scheme):
    root = cfgmod.bm(cfg, "proto_root")
    return os.path.join(root, f"{cfg.general.dataset_name}_text_reliability_{scheme}.json")


def reliability_tensor_path(cfg, scheme):
    root = cfgmod.bm(cfg, "proto_root")
    return os.path.join(root, f"{cfg.general.dataset_name}_class_reliability_{scheme}.pt")


def reliability_metadata_path(cfg, scheme):
    root = cfgmod.bm(cfg, "proto_root")
    return os.path.join(root, f"{cfg.general.dataset_name}_reliability_metadata_{scheme}.json")


def load_selected_text_temperature(cfg, scheme, override=None):
    if override is not None:
        return float(override), "arg"
    sel = os.path.join(utils.method_dir(cfg.general.dataset_name, f"T_{scheme}"),
                       "selected_temperature.json")
    if os.path.exists(sel):
        return float(utils.load_json(sel)["selected_temperature"]), sel
    return 1.0, "default(1.0; run evaluate_text first for tuned Tt)"


def compute_and_save(cfg, scheme, reliability_temperature, text_temperature=None):
    C = cfg.general.num_classes
    Tt, Tt_src = load_selected_text_temperature(cfg, scheme, text_temperature)
    tr_feats, tr_labels = datamod.load_split(cfg, "train")
    datamod.check_cache_integrity(cfg, tr_feats, tr_labels, "train",
                                  datamod.load_extract_meta(cfg))
    protos = datamod.load_text_prototypes(cfg, scheme)
    logit_scale = datamod.get_logit_scale(cfg)
    train_text_logits = logit_scale * (tr_feats @ protos.t())
    rel, class_margin, class_count, margins = estimate_class_text_reliability(
        train_text_logits, tr_labels, C, Tt, reliability_temperature)
    M = mean_text_logits_by_true_class(train_text_logits / float(Tt), tr_labels, C)

    out = reliability_path(cfg, scheme)
    pt_out = reliability_tensor_path(cfg, scheme)
    meta_out = reliability_metadata_path(cfg, scheme)
    metadata = {
        "dataset": cfg.general.dataset_name,
        "text_scheme": scheme,
        "text_temperature": Tt,
        "text_temperature_source": Tt_src,
        "reliability_temperature": reliability_temperature,
        "num_classes": C,
        "source_split": "train",
        "logit_scale": logit_scale,
        "reliability_definition": RELIABILITY_DEFINITION,
    }
    utils.save_json(out, {
        **metadata,
        "reliability": rel.tolist(),
        "raw_margin": class_margin.tolist(),
        "class_count": class_count.tolist(),
        "mean_margin": float(margins.mean().item()),
        "min_margin": float(margins.min().item()),
        "max_margin": float(margins.max().item()),
        "scaled_text_logit_matrix": M.tolist(),
    })
    utils.save_json(meta_out, metadata)
    utils.ensure_dir(os.path.dirname(pt_out))
    torch.save(rel.cpu(), pt_out)
    print(f"[reliability] {scheme}: " +
          " ".join(f"c{c}={rel[c]:.3f}" for c in range(C)))
    print(f"[reliability] saved -> {out}")
    print(f"[reliability] tensor -> {pt_out}")
    return rel


def load_reliability(cfg, scheme, text_temperature=None, allow_legacy=False):
    """Load saved class reliability [C] tensor (raises if absent)."""
    path = reliability_path(cfg, scheme)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run `python -m biomedclip_ltc.text_reliability "
            f"--config <cfg> --text-scheme {scheme}` first.")
    obj = utils.load_json(path)
    if not allow_legacy:
        if obj.get("reliability_definition") != RELIABILITY_DEFINITION:
            raise FileNotFoundError(
                f"{path} was produced by an older reliability definition; recompute it.")
        if text_temperature is not None:
            old_t = float(obj.get("text_temperature", -1.0))
            if abs(old_t - float(text_temperature)) > 1e-6:
                raise FileNotFoundError(
                    f"{path} has Tt={old_t}, expected Tt={text_temperature}; recompute it.")
    return torch.tensor(obj["reliability"], dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--all", action="store_true", help="compute for P0,P1,P2")
    ap.add_argument("--reliability-temperature", type=float, default=None)
    ap.add_argument("--text-temperature", type=float, default=None,
                    help="override Tt; else uses Stage-3 selected_temperature.json")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    temp = (args.reliability_temperature
            if args.reliability_temperature is not None
            else float(cfgmod.bm(cfg, "reliability_temperature")))
    schemes = ["P0", "P1", "P2"] if args.all else [args.text_scheme
                                                   or cfgmod.bm(cfg, "text_scheme")]
    for s in schemes:
        compute_and_save(cfg, s, temp, args.text_temperature)


if __name__ == "__main__":
    main()
