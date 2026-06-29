"""Stage 3 - text-only branch evaluation (no training).

Reads cached image features + class text prototypes and scores images by
    text_logits = logit_scale * (image_features @ text_prototypes.T)
For each text scheme (P0/P1/P2) a temperature T is selected on the VALIDATION set
(final logits = text_logits / T). NOTE: argmax-based metrics (Acc/F1) and per-class
one-vs-rest AUROC/AUPRC are invariant to a positive scale, so T is chosen by lowest
validation cross-entropy (calibration) among the best-accuracy candidates; the
chosen T is what the fusion stage reuses.

Test is evaluated only with --test, after T is fixed on val.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.evaluate_text --config biomedclip_ltc/configs/isic_100.yml --text-scheme P1
    python -m biomedclip_ltc.evaluate_text --config biomedclip_ltc/configs/isic_100.yml --text-scheme P1 --test
"""
import argparse

import torch
import torch.nn.functional as F

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import utils

DEFAULT_TEMPS = [1, 2, 5, 10, 20, 50, 100]


def text_logits(feats, protos, logit_scale):
    return logit_scale * (feats @ protos.t())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--temperatures", default=None,
                    help="comma list, e.g. 1,2,5,10,20,50,100")
    ap.add_argument("--test", action="store_true",
                    help="also run the single final TEST eval (after val selection)")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    scheme = args.text_scheme or cfgmod.bm(cfg, "text_scheme")
    temps = ([float(x) for x in args.temperatures.split(",")]
             if args.temperatures else list(DEFAULT_TEMPS))
    ls = datamod.get_logit_scale(cfg)
    ds = cfg.general.dataset_name
    out = utils.ensure_dir(utils.method_dir(ds, f"T_{scheme}"))
    print(f"[Stage3] text-only ds={ds} scheme={scheme} logit_scale={ls:.3f} "
          f"temps={temps}")

    meta = datamod.load_extract_meta(cfg)
    va_feats, va_labels = datamod.load_split(cfg, "val")
    datamod.check_cache_integrity(cfg, va_feats, va_labels, "val", meta)
    protos = datamod.load_text_prototypes(cfg, scheme)
    base_val = text_logits(va_feats, protos, ls)

    # validation temperature search
    search = []
    best = None
    for T in temps:
        logits = base_val / T
        res = metricsmod.evaluate_logits(cfg, logits, va_labels)
        acc = metricsmod.group_avg_acc(res)
        ce = float(F.cross_entropy(logits, va_labels))
        search.append({"T": T, "group_avg_acc": acc, "val_ce": ce})
        # select by accuracy, tie-break by lower CE (acc is scale-invariant)
        key = (acc, -ce)
        if best is None or key > best[0]:
            best = (key, T, res, ce)
        print(f"  T={T:>5}: " + metricsmod.format_summary(res, "val") + f" | CE={ce:.4f}")
    _, T_sel, va_res, va_ce = best
    print(f"[Stage3] selected T={T_sel} | " + metricsmod.format_summary(va_res, "val(best)"))

    utils.save_json(f"{out}/selected_temperature.json", {
        "text_scheme": scheme, "logit_scale": ls, "selected_temperature": T_sel,
        "val_ce": va_ce, "search": search, "search_space": temps})
    utils.save_json(f"{out}/val_results.json",
                    {"text_scheme": scheme, "temperature": T_sel,
                     "val": metricsmod.results_to_jsonable(va_res)})
    utils.save_json(f"{out}/per_class_val.json",
                    {"temperature": T_sel,
                     "per_class_acc": list(va_res["class_acc"]),
                     "per_class_f1": [float(x) for x in va_res["class_f1"]]})

    if args.test:
        te_feats, te_labels = datamod.load_split(cfg, "test")
        datamod.check_cache_integrity(cfg, te_feats, te_labels, "test", meta)
        te_logits = text_logits(te_feats, protos, ls) / T_sel
        te_res = metricsmod.evaluate_logits(cfg, te_logits, te_labels)
        print(metricsmod.format_summary(te_res, "TEST"))
        utils.save_json(f"{out}/test_results.json",
                        {"text_scheme": scheme, "temperature": T_sel,
                         "test": metricsmod.results_to_jsonable(te_res)})
    print(f"[Stage3] outputs -> {out}")


if __name__ == "__main__":
    main()
