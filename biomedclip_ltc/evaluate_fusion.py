"""Stage 4 & 5 - visual+text fusion with four ablation modes.

    fused_logits = visual_logits / Tv  +  alpha_i * (text_logits / Tt)

modes (selected on VALIDATION only):
    fixed             alpha_i = alpha            (search alpha in [0..1])
    uncertainty_only  alpha_i = lam * u_i        (search lam)
    reliability_only  alpha_i = lam * s_i        (search lam)
    adaptive          alpha_i = lam * u_i * s_i   (search lam)
where
    u_i = normalized entropy of the visual prediction (visual uncertainty),
    s_i = sum_c p_text(c) * class_text_reliability(c)  (sample text reliability).

Inputs are all FROZEN/cached: the Stage-2 visual head checkpoint, Stage-1 text
prototypes, and (for reliability/adaptive) the train-only class reliability.
Class reliability uses TRAIN only; Tt comes from the Stage-3 val selection; alpha/
lambda are chosen on VAL. TEST is evaluated once, with --test, after selection.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.evaluate_fusion --config biomedclip_ltc/configs/isic_100.yml \
        --mode adaptive --text-scheme P1 \
        --visual-run-dir outputs/biomedclip_ltc/isic/V_CE_seed1_lr0.01_bs256_ep50 --test
"""
import argparse
import os

import torch

from biomedclip_ltc import calibration
from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import text_reliability as trmod
from biomedclip_ltc import utils
from biomedclip_ltc.model import VisualHead

DEFAULT_ALPHAS = [round(0.1 * i, 1) for i in range(11)]          # 0.0 .. 1.0
DEFAULT_LAMBDAS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def fuse(visual_logits, text_logits, alpha, Tv, Tt):
    """alpha: scalar or [N] tensor. Returns fused logits [N,C]."""
    a = alpha if not torch.is_tensor(alpha) else alpha.unsqueeze(1)
    return visual_logits / Tv + a * (text_logits / Tt)


def load_text_temperature(cfg, scheme, override):
    if override is not None:
        return float(override), "arg"
    sel = os.path.join(utils.method_dir(cfg.general.dataset_name, f"T_{scheme}"),
                       "selected_temperature.json")
    if os.path.exists(sel):
        return float(utils.load_json(sel)["selected_temperature"]), sel
    return 1.0, "default(1.0; run evaluate_text for a tuned Tt)"


def compute_branch_logits(cfg, head, feats, protos, logit_scale):
    with torch.no_grad():
        visual = head(feats)
    text = logit_scale * (feats @ protos.t())
    return visual, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=list(calibration.MODES))
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--visual-run-dir", required=True,
                    help="Stage-2 run dir containing best.pt")
    ap.add_argument("--visual-temperature", type=float, default=None)
    ap.add_argument("--text-temperature", type=float, default=None,
                    help="override Tt; else taken from Stage-3 selection")
    ap.add_argument("--alphas", default=None, help="comma list for fixed mode")
    ap.add_argument("--lambdas", default=None, help="comma list for adaptive modes")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    scheme = args.text_scheme or cfgmod.bm(cfg, "text_scheme")
    ds = cfg.general.dataset_name
    C = cfg.general.num_classes
    D = int(cfgmod.bm(cfg, "feature_dim"))
    Tv = float(args.visual_temperature if args.visual_temperature is not None
               else cfgmod.bm(cfg, "visual_temperature"))
    Tt, Tt_src = load_text_temperature(cfg, scheme, args.text_temperature)
    ls = datamod.get_logit_scale(cfg)
    device = "cpu"  # fusion is tiny; CPU is fine and deterministic

    # visual head
    ckpt_path = os.path.join(args.visual_run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["feature_dim"] == D and ckpt["num_classes"] == C, \
        "visual checkpoint dims mismatch config"
    seed = ckpt.get("seed", 1)
    head = VisualHead(D, C)
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    # class reliability (train-only) for reliability/adaptive
    class_rel = None
    if args.mode in ("reliability_only", "adaptive"):
        try:
            class_rel = trmod.load_reliability(cfg, scheme)
        except FileNotFoundError:
            print("[fusion] reliability not found; computing from TRAIN now...")
            class_rel = trmod.compute_and_save(
                cfg, scheme, float(cfgmod.bm(cfg, "reliability_temperature")))

    meta = datamod.load_extract_meta(cfg)
    protos = datamod.load_text_prototypes(cfg, scheme)

    def branch(split):
        feats, labels = datamod.load_split(cfg, split)
        datamod.check_cache_integrity(cfg, feats, labels, split, meta)
        v, t = compute_branch_logits(cfg, head, feats, protos, ls)
        u = calibration.visual_uncertainty(v)
        s = (calibration.sample_text_reliability(t / Tt, class_rel)
             if class_rel is not None else None)
        return labels, v, t, u, s

    va_labels, va_v, va_t, va_u, va_s = branch("val")
    n_val = va_labels.shape[0]
    print(f"[fusion] mode={args.mode} ds={ds} scheme={scheme} seed={seed} "
          f"Tv={Tv} Tt={Tt} ({Tt_src}) logit_scale={ls:.3f}")

    # --- validation search ---
    search = []
    best = None  # (acc, key_value, res)
    if args.mode == "fixed":
        grid = ([float(x) for x in args.alphas.split(",")]
                if args.alphas else list(DEFAULT_ALPHAS))
        for a in grid:
            fused = fuse(va_v, va_t, a, Tv, Tt)
            res = metricsmod.evaluate_logits(cfg, fused, va_labels)
            acc = metricsmod.group_avg_acc(res)
            search.append({"alpha": a, "group_avg_acc": acc})
            print(f"  alpha={a:>4}: " + metricsmod.format_summary(res, "val"))
            if best is None or acc > best[0]:
                best = (acc, a, res)
        sel_key, sel_val, va_res = best
        selected = {"alpha": sel_val}
    else:
        grid = ([float(x) for x in args.lambdas.split(",")]
                if args.lambdas else list(DEFAULT_LAMBDAS))
        for lam in grid:
            alpha_i = calibration.fusion_alpha(
                args.mode, lam, uncertainty=va_u, sample_reliability=va_s,
                n=n_val, device=device)
            fused = fuse(va_v, va_t, alpha_i, Tv, Tt)
            res = metricsmod.evaluate_logits(cfg, fused, va_labels)
            acc = metricsmod.group_avg_acc(res)
            search.append({"lambda": lam, "group_avg_acc": acc})
            print(f"  lambda={lam:>4}: " + metricsmod.format_summary(res, "val"))
            if best is None or acc > best[0]:
                best = (acc, lam, res)
        sel_key, sel_val, va_res = best
        selected = {"lambda": sel_val}

    print(f"[fusion] selected {selected} | " +
          metricsmod.format_summary(va_res, "val(best)"))

    out = utils.ensure_dir(utils.method_dir(ds, f"VT_{args.mode}_{scheme}_seed{seed}"))
    utils.save_json(f"{out}/selected_fusion.json", {
        "mode": args.mode, "text_scheme": scheme, "seed": seed,
        "visual_run_dir": args.visual_run_dir,
        "Tv": Tv, "Tt": Tt, "Tt_source": Tt_src, "logit_scale": ls,
        "selected": selected, "search": search,
        "val_group_avg_acc": metricsmod.group_avg_acc(va_res)})
    utils.save_json(f"{out}/val_results.json",
                    {"mode": args.mode, "text_scheme": scheme, **selected,
                     "val": metricsmod.results_to_jsonable(va_res)})
    utils.save_json(f"{out}/per_class_val.json",
                    {**selected, "per_class_acc": list(va_res["class_acc"])})

    # --- single TEST eval (only after selection) ---
    if args.test:
        te_labels, te_v, te_t, te_u, te_s = branch("test")
        if args.mode == "fixed":
            alpha_t = selected["alpha"]
        else:
            alpha_t = calibration.fusion_alpha(
                args.mode, selected["lambda"], uncertainty=te_u,
                sample_reliability=te_s, n=te_labels.shape[0], device=device)
        fused_t = fuse(te_v, te_t, alpha_t, Tv, Tt)
        te_res = metricsmod.evaluate_logits(cfg, fused_t, te_labels)
        print(metricsmod.format_summary(te_res, "TEST"))
        h, m, t, avg = te_res["accuracy"]
        print(f"[fusion][test] overall={avg:.2f} Head={h:.2f} Med={m:.2f} Tail={t:.2f}")
        utils.save_json(f"{out}/test_results.json",
                        {"mode": args.mode, "text_scheme": scheme, **selected,
                         "Tv": Tv, "Tt": Tt,
                         "test": metricsmod.results_to_jsonable(te_res)})
    print(f"[fusion] outputs -> {out}")


if __name__ == "__main__":
    main()
