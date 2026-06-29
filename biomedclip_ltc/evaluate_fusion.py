"""Stage 4 & 5 - visual+text fusion with four ablation modes.

Unified fusion:
    fused_logits = visual_logits / Tv + alpha_i * (text_logits / Tt)

Validation selects Tv and either a fixed alpha or a lambda for the gate. For
uncertainty/reliability/adaptive modes, the raw validation gate mean is saved and
reused unchanged at test time:

    alpha_i = clip(lambda * gate_i / mean_val(gate), 0, alpha_max)

Class reliability is estimated from TRAIN only by text_reliability.py. The test
split can be evaluated either after a validation search with --test (legacy
debug convenience) or strictly with --eval-split test --load-best-config.
"""
import argparse
import csv
import os

import torch

from biomedclip_ltc import calibration
from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import text_reliability as trmod
from biomedclip_ltc import utils
from biomedclip_ltc.model import VisualHead

DEFAULT_ALPHAS = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_LAMBDAS = [round(0.1 * i, 1) for i in range(11)]


def parse_float_grid(value, default):
    if value is None or value == "":
        return list(default)
    return [float(x) for x in value.split(",") if x.strip()]


def fuse(visual_logits, text_logits, alpha, Tv, Tt):
    """alpha: scalar or [N] tensor. Returns fused logits [N,C]."""
    a = alpha if not torch.is_tensor(alpha) else alpha.unsqueeze(1)
    return visual_logits / float(Tv) + a * (text_logits / float(Tt))


def load_text_temperature(cfg, scheme, override):
    if override is not None:
        return float(override), "arg"
    sel = os.path.join(utils.method_dir(cfg.general.dataset_name, f"T_{scheme}"),
                       "selected_temperature.json")
    if os.path.exists(sel):
        return float(utils.load_json(sel)["selected_temperature"]), sel
    return 1.0, "default(1.0; run evaluate_text for a tuned Tt)"


def compute_branch_logits(head, feats, protos, logit_scale):
    with torch.no_grad():
        visual = head(feats)
    text = logit_scale * (feats @ protos.t())
    return visual, text


def load_visual_head(cfg, visual_run_dir):
    C = cfg.general.num_classes
    D = int(cfgmod.bm(cfg, "feature_dim"))
    ckpt_path = os.path.join(visual_run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["feature_dim"] == D and ckpt["num_classes"] == C, \
        "visual checkpoint dims mismatch config"
    head = VisualHead(D, C)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return head, ckpt


def load_split_logits(cfg, split, head, protos, logit_scale, meta):
    feats, labels = datamod.load_split(cfg, split)
    datamod.check_cache_integrity(cfg, feats, labels, split, meta)
    v, t = compute_branch_logits(head, feats, protos, logit_scale)
    return labels, v, t


def compute_signals(visual_logits, text_logits, Tv, Tt, class_rel):
    u = calibration.visual_uncertainty(visual_logits, temperature=Tv)
    s = None
    if class_rel is not None:
        s = calibration.sample_text_reliability(
            text_logits, class_rel, text_temperature=Tt)
    return u, s


def need_reliability(mode):
    return mode in ("reliability_only", "adaptive")


def get_class_reliability(cfg, scheme, Tt, mode, allow_compute):
    if not need_reliability(mode):
        return None
    try:
        return trmod.load_reliability(cfg, scheme, text_temperature=Tt)
    except FileNotFoundError as exc:
        if not allow_compute:
            raise SystemExit(
                f"[fusion] reliability unavailable or stale for {scheme} at Tt={Tt}. "
                "Run `python -m biomedclip_ltc.text_reliability --config <cfg> "
                f"--text-scheme {scheme}` before strict test eval. Details: {exc}")
        print(f"[fusion] reliability missing/stale; recomputing from TRAIN only: {exc}")
        return trmod.compute_and_save(
            cfg, scheme, float(cfgmod.bm(cfg, "reliability_temperature")),
            text_temperature=Tt)


def metric_record(res):
    h, m, t, avg = res["accuracy"]
    _, _, _, auroc = res["aucs"]
    _, _, _, auprc = res["auprcs"]
    _, _, _, f1 = res["f1s"]
    return {
        "group_avg_acc": float(avg),
        "head_acc": float(h),
        "medium_acc": float(m),
        "tail_acc": float(t),
        "auroc": float(100 * auroc),
        "auprc": float(100 * auprc),
        "macro_f1": float(100 * f1),
    }


def alpha_stats(alpha):
    if not torch.is_tensor(alpha):
        return {"alpha_mean": float(alpha), "alpha_min": float(alpha),
                "alpha_max_observed": float(alpha)}
    return {
        "alpha_mean": float(alpha.mean().item()),
        "alpha_min": float(alpha.min().item()),
        "alpha_max_observed": float(alpha.max().item()),
    }


def make_alpha(mode, selected, n, u, s, alpha_max):
    if mode == "fixed":
        alpha, gate_mean = calibration.fusion_alpha(
            mode, selected["alpha"], n=n, alpha_fixed=selected["alpha"],
            alpha_max=alpha_max, return_gate_mean=True)
    else:
        alpha, gate_mean = calibration.fusion_alpha(
            mode, selected["lambda"], uncertainty=u, sample_reliability=s, n=n,
            alpha_max=alpha_max, gate_mean=selected["gate_mean"],
            return_gate_mean=True)
    return alpha, gate_mean


def save_diagnostics(path, labels, visual_logits, text_logits, fused_logits, u, s, alpha):
    utils.ensure_dir(os.path.dirname(path))
    if not torch.is_tensor(alpha):
        alpha = torch.full((labels.shape[0],), float(alpha))
    v_pred = visual_logits.argmax(dim=1)
    t_pred = text_logits.argmax(dim=1)
    f_pred = fused_logits.argmax(dim=1)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "label", "visual_prediction", "text_prediction",
            "fusion_prediction", "visual_uncertainty",
            "sample_text_reliability", "alpha",
        ])
        writer.writeheader()
        for i in range(labels.shape[0]):
            writer.writerow({
                "index": i,
                "label": int(labels[i]),
                "visual_prediction": int(v_pred[i]),
                "text_prediction": int(t_pred[i]),
                "fusion_prediction": int(f_pred[i]),
                "visual_uncertainty": float(u[i]),
                "sample_text_reliability": "" if s is None else float(s[i]),
                "alpha": float(alpha[i]),
            })


def run_eval_split(cfg, split, mode, scheme, selected, head, protos, logit_scale,
                   meta, class_rel, alpha_max, out):
    labels, v, t = load_split_logits(cfg, split, head, protos, logit_scale, meta)
    u, s = compute_signals(v, t, selected["Tv"], selected["Tt"], class_rel)
    alpha, _ = make_alpha(mode, selected, labels.shape[0], u, s, alpha_max)
    fused = fuse(v, t, alpha, selected["Tv"], selected["Tt"])
    res = metricsmod.evaluate_logits(cfg, fused, labels)
    print(metricsmod.format_summary(res, split.upper()))
    save_diagnostics(os.path.join(out, f"{split}_diagnostics.csv"),
                     labels, v, t, fused, u, s, alpha)
    payload = {
        "mode": mode,
        "text_scheme": scheme,
        **selected,
        split: metricsmod.results_to_jsonable(res),
        "alpha_stats": alpha_stats(alpha),
    }
    utils.save_json(os.path.join(out, f"{split}_results.json"), payload)
    return res, alpha


def validation_search(cfg, args, mode, scheme, head, protos, logit_scale, meta,
                      class_rel, Tt, Tt_src, seed, visual_run_dir, alpha_max):
    Tv_base = float(args.visual_temperature if args.visual_temperature is not None
                    else cfgmod.bm(cfg, "visual_temperature"))
    Tv_grid = parse_float_grid(args.visual_temperatures, [Tv_base])
    alphas = parse_float_grid(args.alphas, DEFAULT_ALPHAS)
    lambdas = parse_float_grid(args.lambdas, DEFAULT_LAMBDAS)

    labels, va_v, va_t = load_split_logits(cfg, "val", head, protos, logit_scale, meta)
    n_val = labels.shape[0]
    print(f"[fusion] mode={mode} ds={cfg.general.dataset_name} scheme={scheme} "
          f"seed={seed} Tv_grid={Tv_grid} Tt={Tt} ({Tt_src}) "
          f"alpha_max={alpha_max} logit_scale={logit_scale:.3f}")

    search = []
    best = None
    for Tv in Tv_grid:
        va_u, va_s = compute_signals(va_v, va_t, Tv, Tt, class_rel)
        if mode == "fixed":
            for a in alphas:
                selected = {"alpha": a, "lambda": a, "Tv": Tv, "Tt": Tt,
                            "gate_mean": 1.0, "alpha_max": alpha_max}
                alpha, _ = make_alpha(mode, selected, n_val, va_u, va_s, alpha_max)
                fused = fuse(va_v, va_t, alpha, Tv, Tt)
                res = metricsmod.evaluate_logits(cfg, fused, labels)
                rec = {"Tv": Tv, "Tt": Tt, "alpha": a, **metric_record(res),
                       **alpha_stats(alpha)}
                search.append(rec)
                print(f"  Tv={Tv:g} alpha={a:>4}: " +
                      metricsmod.format_summary(res, "val"))
                key = rec["group_avg_acc"]
                if best is None or key > best[0]:
                    best = (key, selected, res, alpha, va_u, va_s, fused)
        else:
            for lam in lambdas:
                alpha, gate_mean = calibration.fusion_alpha(
                    mode, lam, uncertainty=va_u, sample_reliability=va_s, n=n_val,
                    alpha_max=alpha_max, gate_mean=None, return_gate_mean=True)
                fused = fuse(va_v, va_t, alpha, Tv, Tt)
                res = metricsmod.evaluate_logits(cfg, fused, labels)
                rec = {"Tv": Tv, "Tt": Tt, "lambda": lam,
                       "gate_mean": gate_mean, **metric_record(res),
                       **alpha_stats(alpha)}
                search.append(rec)
                print(f"  Tv={Tv:g} lambda={lam:>4} gate_mean={gate_mean:.4f}: " +
                      metricsmod.format_summary(res, "val"))
                key = rec["group_avg_acc"]
                selected = {"lambda": lam, "Tv": Tv, "Tt": Tt,
                            "gate_mean": gate_mean, "alpha_max": alpha_max}
                if best is None or key > best[0]:
                    best = (key, selected, res, alpha, va_u, va_s, fused)

    _, selected, va_res, va_alpha, va_u, va_s, va_fused = best
    print(f"[fusion] selected {selected} | " +
          metricsmod.format_summary(va_res, "val(best)"))

    out = utils.ensure_dir(utils.method_dir(
        cfg.general.dataset_name, f"VT_{mode}_{scheme}_seed{seed}"))
    utils.save_json(os.path.join(out, "selected_fusion.json"), {
        "mode": mode,
        "text_scheme": scheme,
        "seed": seed,
        "visual_run_dir": visual_run_dir,
        "Tv": selected["Tv"],
        "Tt": selected["Tt"],
        "Tt_source": Tt_src,
        "logit_scale": logit_scale,
        "alpha_max": alpha_max,
        "selected": selected,
        "search": search,
        "val_group_avg_acc": metricsmod.group_avg_acc(va_res),
        "alpha_stats": alpha_stats(va_alpha),
    })
    utils.save_json(os.path.join(out, "val_results.json"), {
        "mode": mode,
        "text_scheme": scheme,
        **selected,
        "val": metricsmod.results_to_jsonable(va_res),
        "alpha_stats": alpha_stats(va_alpha),
    })
    utils.save_json(os.path.join(out, "per_class_val.json"), {
        **selected,
        "per_class_acc": list(va_res["class_acc"]),
    })
    save_diagnostics(os.path.join(out, "val_diagnostics.csv"),
                     labels, va_v, va_t, va_fused, va_u, va_s, va_alpha)
    return selected, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default=None, choices=list(calibration.MODES))
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--visual-run-dir", default=None,
                    help="Stage-2 run dir containing best.pt")
    ap.add_argument("--visual-temperature", type=float, default=None)
    ap.add_argument("--visual-temperatures", default=None,
                    help="comma list of Tv values for validation search")
    ap.add_argument("--text-temperature", type=float, default=None,
                    help="override Tt; else taken from Stage-3 selection")
    ap.add_argument("--alphas", default=None, help="comma list for fixed mode")
    ap.add_argument("--lambdas", default=None, help="comma list for gated modes")
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--load-best-config", default=None,
                    help="path to selected_fusion.json for strict test eval")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--test", action="store_true",
                    help="legacy debug: run test once after validation selection")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    loaded = utils.load_json(args.load_best_config) if args.load_best_config else None
    mode = args.mode or (loaded or {}).get("mode")
    if mode is None:
        raise SystemExit("--mode is required unless --load-best-config is provided")
    scheme = args.text_scheme or (loaded or {}).get("text_scheme") or cfgmod.bm(cfg, "text_scheme")
    visual_run_dir = args.visual_run_dir or (loaded or {}).get("visual_run_dir")
    if visual_run_dir is None:
        raise SystemExit("--visual-run-dir is required unless saved in --load-best-config")

    if loaded is not None:
        selected = dict(loaded["selected"])
        selected.setdefault("Tv", loaded.get("Tv", selected.get("Tv", 1.0)))
        selected.setdefault("Tt", loaded.get("Tt", selected.get("Tt", 1.0)))
        selected.setdefault("alpha_max", loaded.get("alpha_max", args.alpha_max))
        selected.setdefault("gate_mean", 1.0)
        alpha_max = float(selected.get("alpha_max", args.alpha_max))
        Tt, Tt_src = float(selected["Tt"]), args.load_best_config
    else:
        selected = None
        alpha_max = float(args.alpha_max)
        Tt, Tt_src = load_text_temperature(cfg, scheme, args.text_temperature)

    head, ckpt = load_visual_head(cfg, visual_run_dir)
    seed = ckpt.get("seed", 1)
    meta = datamod.load_extract_meta(cfg)
    protos = datamod.load_text_prototypes(cfg, scheme)
    logit_scale = datamod.get_logit_scale(cfg)

    strict_test = args.eval_split == "test" and loaded is not None
    class_rel = get_class_reliability(
        cfg, scheme, Tt, mode, allow_compute=not strict_test)

    if args.eval_split == "test":
        if loaded is None:
            raise SystemExit("--eval-split test requires --load-best-config; "
                             "test mode does not search hyper-parameters")
        out = utils.ensure_dir(os.path.dirname(args.load_best_config))
        run_eval_split(cfg, "test", mode, scheme, selected, head, protos,
                       logit_scale, meta, class_rel, alpha_max, out)
        print(f"[fusion] outputs -> {out}")
        return

    selected, out = validation_search(
        cfg, args, mode, scheme, head, protos, logit_scale, meta, class_rel,
        Tt, Tt_src, seed, visual_run_dir, alpha_max)

    if args.test:
        run_eval_split(cfg, "test", mode, scheme, selected, head, protos,
                       logit_scale, meta, class_rel, alpha_max, out)
    print(f"[fusion] outputs -> {out}")


if __name__ == "__main__":
    main()
