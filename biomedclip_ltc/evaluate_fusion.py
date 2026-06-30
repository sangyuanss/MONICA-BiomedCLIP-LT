"""Stage 4 & 5 - visual+text fusion with logit/probability ablation modes.

Unified fusion:
    fused_logits = visual_logits / Tv + alpha_i * (text_logits / Tt)

Validation selects Tv and either a fixed alpha or a lambda for the gate. For
uncertainty/reliability/adaptive modes, the raw validation gate mean is saved and
reused unchanged at test time:

    alpha_i = clip(lambda * gate_i / mean_val(gate), 0, alpha_max)

Class reliability is estimated from TRAIN only by text_reliability.py. The test
split can be evaluated either after a validation search with --test (legacy
debug convenience) or strictly with --eval-split test --load-best-config.

Probability-domain class-channel modes mix calibrated visual/text probabilities
with a sample-class gate:

    a_ic = alpha_max * G_i * B_c * Q_c * R_c
    p_hat_ic = (1 - a_ic) * p_v_ic + a_ic * p_t_ic
    p_fuse = normalize(p_hat)

The sample-benefit diagnostic modes first convert B_c to a scalar sample text
confidence S_i = sum_c p_text_ic * B_c, then use a sample-level gate [N,1].
Old Q/R and class-channel B modes stay available for reproducing previous
ablations.
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F

from biomedclip_ltc import calibration
from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import text_benefit as tbmod
from biomedclip_ltc import text_reliability as trmod
from biomedclip_ltc import utils
from biomedclip_ltc.model import VisualHead

DEFAULT_ALPHAS = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_PROB_ALPHA_MAXES = [0.0, 0.1, 0.2, 0.3, 0.4]
DEFAULT_GAMMAS = [0.0, 0.25, 0.5, 1.0]
DEFAULT_ETAS = [0.0, 0.5, 1.0]
DEFAULT_TAU_BS = [0.25, 0.5, 1.0]
PROB_MODES = (
    "prob_fixed", "prob_G", "prob_B", "prob_GB",
    "prob_B_sample", "prob_GB_sample",
    "prob_Q", "prob_R", "prob_GQ", "prob_GR", "prob_QR", "prob_GQR",
)
ALL_MODES = tuple(calibration.MODES) + PROB_MODES


def parse_float_grid(value, default):
    if value is None or value == "":
        return list(default)
    return [float(x) for x in value.split(",") if x.strip()]


def parse_two_floats(value, default):
    vals = parse_float_grid(value, default)
    if len(vals) != 2:
        raise ValueError(f"expected two comma-separated values, got {value}")
    lo, hi = vals
    if not 0 <= lo < hi <= 1:
        raise ValueError(f"quantiles must satisfy 0 <= low < high <= 1, got {vals}")
    return lo, hi


def is_prob_mode(mode):
    return mode in PROB_MODES


def prob_uses_gate(mode, name):
    return name in mode.replace("prob_", "")


def need_benefit(mode):
    return is_prob_mode(mode) and prob_uses_gate(mode, "B")


def is_sample_benefit_mode(mode):
    return mode in ("prob_B_sample", "prob_GB_sample")


def benefit_vector_from_obj(benefit_obj, tau_B):
    return tbmod.benefit_from_conservative_gain(
        benefit_obj["class_conservative_gain"], tau_B)


def fuse(visual_logits, text_logits, alpha, Tv, Tt):
    """alpha: scalar or [N] tensor. Returns fused logits [N,C]."""
    a = alpha if not torch.is_tensor(alpha) else alpha.unsqueeze(1)
    return visual_logits / float(Tv) + a * (text_logits / float(Tt))


def fuse_probabilities_with_class_gate(visual_probs, text_probs, alpha_ic,
                                       eps=1e-8):
    """Probability-domain fusion with one gate per sample and class."""
    if visual_probs.shape != text_probs.shape:
        raise ValueError(
            "visual_probs and text_probs must have identical shapes, "
            f"got {visual_probs.shape} and {text_probs.shape}")
    if not torch.is_tensor(alpha_ic):
        alpha_ic = torch.full_like(visual_probs, float(alpha_ic))
    elif alpha_ic.dim() == 1:
        alpha_ic = alpha_ic[:, None].expand_as(visual_probs)
    alpha_ic = alpha_ic.to(visual_probs.device, dtype=visual_probs.dtype).clamp(0.0, 1.0)
    p_hat = (1.0 - alpha_ic) * visual_probs + alpha_ic * text_probs
    p_fuse = p_hat / p_hat.sum(dim=1, keepdim=True).clamp_min(eps)
    return p_fuse


def prob_scores(visual_logits, text_logits, alpha_ic, Tv, Tt, eps=1e-8):
    """Probability-domain fusion, returned as log-prob scores for MONICA metrics."""
    p_v = F.softmax(visual_logits / float(Tv), dim=1)
    p_t = F.softmax(text_logits / float(Tt), dim=1)
    p_fuse = fuse_probabilities_with_class_gate(p_v, p_t, alpha_ic, eps)
    return p_fuse.clamp_min(eps).log()


def compute_sample_text_benefit(text_probs, class_benefit):
    """Convert class-level benefit B_c into sample-level text confidence S_i."""
    if text_probs.ndim != 2:
        raise ValueError(f"text_probs must have shape [N, C], got {text_probs.shape}")
    if class_benefit.ndim != 1:
        raise ValueError(
            f"class_benefit must have shape [C], got {class_benefit.shape}")
    if text_probs.shape[1] != class_benefit.shape[0]:
        raise ValueError(
            "Class count mismatch: "
            f"text_probs has {text_probs.shape[1]} classes, "
            f"class_benefit has {class_benefit.shape[0]}")
    benefit = class_benefit.to(
        device=text_probs.device, dtype=text_probs.dtype).view(1, -1)
    sample_confidence = (text_probs * benefit).sum(dim=1, keepdim=True)
    return sample_confidence.clamp(0.0, 1.0)


def fuse_probabilities_with_sample_gate(visual_probs, text_probs, sample_alpha,
                                        eps=1e-12):
    """Probability-domain fusion with one scalar gate per sample."""
    if visual_probs.shape != text_probs.shape:
        raise ValueError(
            "visual_probs and text_probs must have identical shapes, "
            f"got {visual_probs.shape} and {text_probs.shape}")
    if sample_alpha.ndim == 1:
        sample_alpha = sample_alpha.view(-1, 1)
    expected_shape = (visual_probs.shape[0], 1)
    if tuple(sample_alpha.shape) != expected_shape:
        raise ValueError(
            f"sample_alpha must have shape {expected_shape}, "
            f"got {tuple(sample_alpha.shape)}")
    sample_alpha = sample_alpha.to(
        device=visual_probs.device, dtype=visual_probs.dtype).clamp(0.0, 1.0)
    fused = (1.0 - sample_alpha) * visual_probs + sample_alpha * text_probs
    fused = fused.clamp_min(eps)
    return fused / fused.sum(dim=1, keepdim=True).clamp_min(eps)


def uncertainty_components(visual_logits, temperature=1.0, eps=1e-8):
    p = F.softmax(visual_logits / float(temperature), dim=1)
    ent = -(p * p.clamp_min(eps).log()).sum(dim=1)
    ent = (ent / torch.log(torch.tensor(float(p.shape[1]), device=p.device))).clamp(0.0, 1.0)
    top2 = p.topk(2, dim=1).values
    margin_u = (1.0 - (top2[:, 0] - top2[:, 1])).clamp(0.0, 1.0)
    return ent, margin_u


def combined_uncertainty(visual_logits, temperature=1.0, eta=1.0):
    ent, margin_u = uncertainty_components(visual_logits, temperature)
    return (float(eta) * ent + (1.0 - float(eta)) * margin_u).clamp(0.0, 1.0)


def map_uncertainty_gate(uncertainty, tau_low, tau_high, eps=1e-8):
    return ((uncertainty - float(tau_low)) / (float(tau_high) - float(tau_low) + eps)).clamp(0.0, 1.0)


def fit_uncertainty_gate(uncertainty, low_q=0.3, high_q=0.8):
    tau_low = float(torch.quantile(uncertainty, float(low_q)).item())
    tau_high = float(torch.quantile(uncertainty, float(high_q)).item())
    if tau_high <= tau_low:
        tau_high = tau_low + 1e-6
    return map_uncertainty_gate(uncertainty, tau_low, tau_high), tau_low, tau_high


def class_tail_need(cfg, gamma, eps=1e-6):
    """Q_c in [0,1]: long-tail compensation need from TRAIN counts only."""
    _, train_labels = datamod.load_split(cfg, "train")
    counts = torch.bincount(train_labels, minlength=cfg.general.num_classes).float()
    need = (counts + eps).pow(-float(gamma))
    return need / need.max().clamp_min(eps)


def build_prob_alpha(mode, n, C, alpha_max, gate=None, tail_need=None,
                     class_rel=None, benefit=None, device=None):
    alpha = torch.full((n, C), float(alpha_max), device=device)
    if prob_uses_gate(mode, "G"):
        if gate is None:
            raise ValueError(f"{mode} requires sample uncertainty gate G")
        alpha = alpha * gate.to(device)[:, None]
    if prob_uses_gate(mode, "B"):
        if benefit is None:
            raise ValueError(f"{mode} requires class text-benefit vector B")
        alpha = alpha * benefit.to(device)[None, :]
    if prob_uses_gate(mode, "Q"):
        if tail_need is None:
            raise ValueError(f"{mode} requires class tail-need vector Q")
        alpha = alpha * tail_need.to(device)[None, :]
    if prob_uses_gate(mode, "R"):
        if class_rel is None:
            raise ValueError(f"{mode} requires class text reliability R")
        alpha = alpha * class_rel.to(device)[None, :]
    return alpha.clamp(0.0, float(alpha_max))


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
    return mode in ("reliability_only", "adaptive") or (
        is_prob_mode(mode) and prob_uses_gate(mode, "R"))


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


def get_text_benefit(cfg, scheme, visual_run_dir, ckpt, mode, benefit_file):
    if not need_benefit(mode):
        return None
    path = benefit_file or tbmod.benefit_path(os.path.abspath(visual_run_dir), scheme)
    obj = tbmod.load_benefit(path)
    tbmod.validate_benefit(obj, cfg, visual_run_dir, scheme, ckpt=ckpt)
    return obj


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


def assert_prob_zero_alpha_consistency(p_fused, p_v, rec, visual_prob_rec,
                                       atol=1e-7, rtol=1e-6):
    torch.testing.assert_close(p_fused, p_v, rtol=rtol, atol=atol)
    for key in ("auroc", "auprc", "group_avg_acc", "macro_f1"):
        diff = abs(float(rec[key]) - float(visual_prob_rec[key]))
        assert diff < 1e-8, (
            f"alpha_max=0 probability baseline mismatch for {key}: "
            f"candidate={rec[key]} visual_prob={visual_prob_rec[key]} diff={diff}")


def alpha_stats(alpha):
    if not torch.is_tensor(alpha):
        return {"alpha_mean": float(alpha), "alpha_min": float(alpha),
                "alpha_max_observed": float(alpha)}
    return {
        "alpha_mean": float(alpha.mean().item()),
        "alpha_min": float(alpha.min().item()),
        "alpha_max_observed": float(alpha.max().item()),
    }


def summarize_tensor(x):
    x = x.detach().float().cpu().view(-1)
    return {
        "min": float(x.min()),
        "q25": float(torch.quantile(x, 0.25)),
        "median": float(torch.quantile(x, 0.50)),
        "mean": float(x.mean()),
        "q75": float(torch.quantile(x, 0.75)),
        "max": float(x.max()),
        "nonzero_ratio": float((x > 1e-8).float().mean()),
    }


def prob_summary_payload(benefit=None, sample_confidence=None, gate=None,
                         alpha=None):
    payload = {}
    if benefit is not None:
        payload["B_summary"] = summarize_tensor(benefit)
    if sample_confidence is not None:
        payload["S_summary"] = summarize_tensor(sample_confidence)
    if gate is not None:
        payload["G_summary"] = summarize_tensor(gate)
    if alpha is not None:
        payload["alpha_summary"] = summarize_tensor(alpha)
    return payload


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


def save_diagnostics(path, labels, visual_logits, text_logits, fused_logits, u, s,
                     alpha, gate=None, tail_need=None, class_rel=None,
                     benefit=None, visual_entropy=None, visual_margin=None,
                     sample_text_confidence=None, text_probs=None):
    utils.ensure_dir(os.path.dirname(path))
    if not torch.is_tensor(alpha):
        alpha = torch.full((labels.shape[0],), float(alpha))
    v_pred = visual_logits.argmax(dim=1)
    t_pred = text_logits.argmax(dim=1)
    f_pred = fused_logits.argmax(dim=1)
    alpha_is_class_matrix = alpha.dim() == 2 and alpha.shape[1] > 1
    if visual_entropy is None or visual_margin is None:
        visual_entropy, visual_margin = uncertainty_components(visual_logits)
    if text_probs is None:
        text_probs = F.softmax(text_logits, dim=1)
    text_top = text_probs.argmax(dim=1)
    text_top_prob = text_probs.gather(1, text_top[:, None]).squeeze(1)
    benefit_for_diag = benefit.to(labels.device) if benefit is not None else None
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_index", "index", "label",
            "visual_pred", "text_pred", "fused_pred",
            "visual_prediction", "text_prediction", "fusion_prediction",
            "visual_correct", "text_correct", "fused_correct",
            "G", "B_text_top", "B_visual_top", "B_fused_top",
            "B_true", "text_top_class", "text_top_probability",
            "sample_text_confidence", "sample_alpha",
            "alpha_mean", "alpha_max_sample", "visual_entropy", "visual_margin",
            "visual_uncertainty",
            "sample_text_reliability", "visual_gate",
            "tail_need_fusion_prediction", "text_reliability_fusion_prediction",
            "alpha", "alpha_fusion_prediction",
        ])
        writer.writeheader()
        for i in range(labels.shape[0]):
            fp = int(f_pred[i])
            if alpha_is_class_matrix:
                alpha_mean = float(alpha[i].mean())
                alpha_fp = float(alpha[i, fp])
                alpha_max_sample = float(alpha[i].max())
                sample_alpha = alpha_mean
            elif alpha.dim() == 2:
                alpha_mean = float(alpha[i, 0])
                alpha_fp = float(alpha[i, 0])
                alpha_max_sample = float(alpha[i, 0])
                sample_alpha = float(alpha[i, 0])
            else:
                alpha_mean = float(alpha[i])
                alpha_fp = float(alpha[i])
                alpha_max_sample = float(alpha[i])
                sample_alpha = float(alpha[i])
            txt_top = int(text_top[i])
            writer.writerow({
                "sample_index": i,
                "index": i,
                "label": int(labels[i]),
                "visual_pred": int(v_pred[i]),
                "text_pred": int(t_pred[i]),
                "fused_pred": fp,
                "visual_prediction": int(v_pred[i]),
                "text_prediction": int(t_pred[i]),
                "fusion_prediction": fp,
                "visual_correct": int(v_pred[i] == labels[i]),
                "text_correct": int(t_pred[i] == labels[i]),
                "fused_correct": int(f_pred[i] == labels[i]),
                "G": "" if gate is None else float(gate[i]),
                "B_text_top": "" if benefit_for_diag is None else float(benefit_for_diag[txt_top]),
                "B_visual_top": "" if benefit_for_diag is None else float(benefit_for_diag[int(v_pred[i])]),
                "B_fused_top": "" if benefit_for_diag is None else float(benefit_for_diag[fp]),
                "B_true": "" if benefit_for_diag is None else float(benefit_for_diag[int(labels[i])]),
                "text_top_class": txt_top,
                "text_top_probability": float(text_top_prob[i]),
                "sample_text_confidence": "" if sample_text_confidence is None
                else float(sample_text_confidence[i].view(-1)[0]),
                "sample_alpha": sample_alpha,
                "alpha_mean": alpha_mean,
                "alpha_max_sample": alpha_max_sample,
                "visual_entropy": float(visual_entropy[i]),
                "visual_margin": float(visual_margin[i]),
                "visual_uncertainty": float(u[i]),
                "sample_text_reliability": "" if s is None else float(s[i]),
                "visual_gate": "" if gate is None else float(gate[i]),
                "tail_need_fusion_prediction": "" if tail_need is None else float(tail_need[fp]),
                "text_reliability_fusion_prediction": "" if class_rel is None else float(class_rel[fp]),
                "alpha": alpha_mean,
                "alpha_fusion_prediction": alpha_fp,
            })


def run_eval_split(cfg, split, mode, scheme, selected, head, protos, logit_scale,
                   meta, class_rel, alpha_max, out, benefit_obj=None):
    if is_prob_mode(mode):
        return run_prob_eval_split(cfg, split, mode, scheme, selected, head, protos,
                                   logit_scale, meta, class_rel, out, benefit_obj)
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


def run_prob_eval_split(cfg, split, mode, scheme, selected, head, protos,
                        logit_scale, meta, class_rel, out, benefit_obj=None):
    labels, v, t = load_split_logits(cfg, split, head, protos, logit_scale, meta)
    ent_u, margin_u = uncertainty_components(v, selected["Tv"])
    U = combined_uncertainty(v, selected["Tv"], selected.get("eta", 1.0))
    gate = None
    if prob_uses_gate(mode, "G"):
        gate = map_uncertainty_gate(U, selected["tau_low"], selected["tau_high"])
    tail_need = None
    if prob_uses_gate(mode, "Q"):
        tail_need = class_tail_need(cfg, selected.get("gamma", 0.0))
    benefit = None
    if prob_uses_gate(mode, "B"):
        if benefit_obj is None:
            benefit_obj = tbmod.load_benefit(selected["benefit_file"])
        benefit = benefit_vector_from_obj(benefit_obj, selected["tau_B"])
    p_v = F.softmax(v / float(selected["Tv"]), dim=1)
    p_t = F.softmax(t / float(selected["Tt"]), dim=1)
    sample_confidence = None
    if is_sample_benefit_mode(mode) or selected.get("benefit_application") == "sample_expectation":
        sample_confidence = compute_sample_text_benefit(p_t, benefit)
        alpha = float(selected["alpha_max"]) * sample_confidence
        if prob_uses_gate(mode, "G"):
            alpha = alpha * gate.to(v.device)[:, None]
        p_fused = fuse_probabilities_with_sample_gate(p_v, p_t, alpha)
        scores = p_fused.clamp_min(1e-12).log()
    else:
        alpha = build_prob_alpha(
            mode, labels.shape[0], cfg.general.num_classes, selected["alpha_max"],
            gate=gate, tail_need=tail_need, class_rel=class_rel,
            benefit=benefit, device=v.device)
        scores = prob_scores(v, t, alpha, selected["Tv"], selected["Tt"])
    res = metricsmod.evaluate_logits(cfg, scores, labels)
    print(metricsmod.format_summary(res, split.upper()))
    sample_rel = (calibration.sample_text_reliability(t, class_rel, selected["Tt"])
                  if class_rel is not None else None)
    summaries = prob_summary_payload(benefit, sample_confidence, gate, alpha)
    save_diagnostics(os.path.join(out, f"{split}_diagnostics.csv"),
                     labels, v, t, scores, U, sample_rel, alpha,
                     gate=gate, tail_need=tail_need, class_rel=class_rel,
                     benefit=benefit, visual_entropy=ent_u, visual_margin=margin_u,
                     sample_text_confidence=sample_confidence, text_probs=p_t)
    if benefit_obj is not None and benefit is not None:
        save_benefit_class_diagnostics(
            os.path.join(out, f"{split}_class_diagnostics.csv"),
            cfg, benefit_obj, benefit, labels, v, scores, split)
    payload = {
        "mode": mode,
        "fusion_domain": "prob",
        "text_scheme": scheme,
        **selected,
        split: metricsmod.results_to_jsonable(res),
        "alpha_stats": alpha_stats(alpha),
        **summaries,
    }
    utils.save_json(os.path.join(out, f"{split}_results.json"), payload)
    return res, alpha


def passes_selection_constraints(rec, visual_rec, auroc_drop, auprc_drop):
    return (rec["auroc"] >= visual_rec["auroc"] - float(auroc_drop)
            and rec["auprc"] >= visual_rec["auprc"] - float(auprc_drop))


def add_selection_constraint_fields(rec, baseline_rec, auroc_drop, auprc_drop,
                                    baseline_domain):
    rec["selection_baseline_domain"] = baseline_domain
    rec["selection_baseline_auroc"] = baseline_rec["auroc"]
    rec["selection_baseline_auprc"] = baseline_rec["auprc"]
    rec["selection_auroc_threshold"] = baseline_rec["auroc"] - float(auroc_drop)
    rec["selection_auprc_threshold"] = baseline_rec["auprc"] - float(auprc_drop)
    rec["constraint_loss"] = constraint_loss(rec, baseline_rec)
    rec["passes_constraints"] = passes_selection_constraints(
        rec, baseline_rec, auroc_drop, auprc_drop)
    return rec


def selection_key(rec):
    if not rec.get("passes_constraints", True):
        return (0, -float(rec.get("constraint_loss", 0.0)),
                rec["group_avg_acc"], rec["tail_acc"], rec["macro_f1"],
                rec["auprc"], rec["auroc"], -float(rec.get("alpha_max", 0.0)))
    return (1 if rec.get("passes_constraints", True) else 0,
            rec["group_avg_acc"], rec["tail_acc"], rec["macro_f1"],
            rec["auprc"], rec["auroc"], -float(rec.get("alpha_max", 0.0)))


def constraint_loss(rec, visual_rec):
    return max(0.0, visual_rec["auroc"] - rec["auroc"]) + \
        max(0.0, visual_rec["auprc"] - rec["auprc"])


def per_class_accuracy(pred, labels, num_classes):
    out = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            out.append(float((pred[mask] == labels[mask]).float().mean().item() * 100.0))
        else:
            out.append(0.0)
    return out


def save_benefit_class_diagnostics(path, cfg, benefit_obj, B, labels,
                                   visual_scores, fused_scores, split):
    utils.ensure_dir(os.path.dirname(path))
    C = cfg.general.num_classes
    visual_pred = visual_scores.argmax(dim=1)
    fused_pred = fused_scores.argmax(dim=1)
    split_visual_acc = per_class_accuracy(visual_pred, labels, C)
    split_fused_acc = per_class_accuracy(fused_pred, labels, C)
    field_split_visual = f"{split}_visual_accuracy"
    field_split_fused = f"{split}_fused_accuracy"
    field_split_gain = f"{split}_accuracy_gain"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_id", "class_count", "mean_gain", "std_gain",
            "standard_error", "conservative_gain", "B_selected",
            "visual_oof_accuracy", "text_accuracy",
            "val_visual_accuracy", "val_fused_accuracy", "val_accuracy_gain",
            field_split_visual, field_split_fused, field_split_gain,
        ])
        writer.writeheader()
        for c in range(C):
            row = {
                "class_id": c,
                "class_count": benefit_obj["class_counts"][c],
                "mean_gain": benefit_obj["class_mean_gain"][c],
                "std_gain": benefit_obj["class_std_gain"][c],
                "standard_error": benefit_obj["class_standard_error"][c],
                "conservative_gain": benefit_obj["class_conservative_gain"][c],
                "B_selected": float(B[c]),
                "visual_oof_accuracy": benefit_obj["class_visual_oof_accuracy"][c],
                "text_accuracy": benefit_obj["class_text_accuracy"][c],
                "val_visual_accuracy": split_visual_acc[c] if split == "val" else "",
                "val_fused_accuracy": split_fused_acc[c] if split == "val" else "",
                "val_accuracy_gain": (split_fused_acc[c] - split_visual_acc[c]
                                      if split == "val" else ""),
                field_split_visual: split_visual_acc[c],
                field_split_fused: split_fused_acc[c],
                field_split_gain: split_fused_acc[c] - split_visual_acc[c],
            }
            writer.writerow(row)


def validation_search(cfg, args, mode, scheme, head, protos, logit_scale, meta,
                      class_rel, Tt, Tt_src, seed, visual_run_dir, alpha_max,
                      benefit_obj=None):
    if is_prob_mode(mode):
        return prob_validation_search(cfg, args, mode, scheme, head, protos,
                                      logit_scale, meta, class_rel, Tt, Tt_src,
                                      seed, visual_run_dir, benefit_obj)

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


def prob_validation_search(cfg, args, mode, scheme, head, protos, logit_scale,
                           meta, class_rel, Tt, Tt_src, seed, visual_run_dir,
                           benefit_obj=None):
    Tv_base = float(args.visual_temperature if args.visual_temperature is not None
                    else cfgmod.bm(cfg, "visual_temperature"))
    Tv_grid = parse_float_grid(args.visual_temperatures, [Tv_base])
    alpha_maxes = parse_float_grid(args.alpha_maxes, DEFAULT_PROB_ALPHA_MAXES)
    gammas = parse_float_grid(args.gammas, DEFAULT_GAMMAS)
    etas = parse_float_grid(args.etas, DEFAULT_ETAS)
    tau_Bs = parse_float_grid(args.tau_Bs, DEFAULT_TAU_BS)
    q_low, q_high = parse_two_floats(args.uncertainty_quantiles, [0.3, 0.8])

    uses_G = prob_uses_gate(mode, "G")
    uses_B = prob_uses_gate(mode, "B")
    uses_Q = prob_uses_gate(mode, "Q")
    uses_R = prob_uses_gate(mode, "R")
    uses_sample_B = is_sample_benefit_mode(mode)
    if uses_B and benefit_obj is None:
        raise SystemExit(f"{mode} requires a text benefit file. Run text_benefit first.")
    benefit_file = None
    if uses_B:
        benefit_file = args.benefit_file or tbmod.benefit_path(
            os.path.abspath(visual_run_dir), scheme)
    gamma_grid = gammas if uses_Q else [0.0]
    eta_grid = etas if uses_G else [1.0]
    tau_B_grid = tau_Bs if uses_B else [None]

    labels, va_v, va_t = load_split_logits(cfg, "val", head, protos, logit_scale, meta)
    n_val, C = labels.shape[0], cfg.general.num_classes
    print(f"[fusion] mode={mode} ds={cfg.general.dataset_name} scheme={scheme} "
          f"seed={seed} domain=prob Tv_grid={Tv_grid} Tt={Tt} ({Tt_src}) "
          f"alpha_maxes={alpha_maxes} gammas={gamma_grid} etas={eta_grid} "
          f"tau_Bs={tau_B_grid} "
          f"q=({q_low},{q_high}) logit_scale={logit_scale:.3f}")

    search = []
    best = None
    for Tv in Tv_grid:
        p_v = F.softmax(va_v / float(Tv), dim=1)
        p_t = F.softmax(va_t / float(Tt), dim=1)
        visual_logit_res = metricsmod.evaluate_logits(cfg, va_v / float(Tv), labels)
        visual_logit_rec = metric_record(visual_logit_res)
        visual_prob_scores = p_v.clamp_min(1e-8).log()
        visual_prob_res = metricsmod.evaluate_logits(cfg, visual_prob_scores, labels)
        visual_prob_rec = metric_record(visual_prob_res)
        print("  baseline " +
              metricsmod.format_summary(visual_logit_res, "visual-logit-val"))
        print("  baseline " +
              metricsmod.format_summary(visual_prob_res, f"visual-prob-val Tv={Tv:g}"))
        print("  selection thresholds: "
              f"AUROC >= {visual_prob_rec['auroc']:.4f} - {args.selection_auroc_drop:g} "
              f"({visual_prob_rec['auroc'] - args.selection_auroc_drop:.4f}); "
              f"AUPRC >= {visual_prob_rec['auprc']:.4f} - {args.selection_auprc_drop:g} "
              f"({visual_prob_rec['auprc'] - args.selection_auprc_drop:.4f})")
        for eta in eta_grid:
            U = combined_uncertainty(va_v, Tv, eta)
            if uses_G:
                G, tau_low, tau_high = fit_uncertainty_gate(U, q_low, q_high)
            else:
                G, tau_low, tau_high = None, None, None
            for gamma in gamma_grid:
                Q = class_tail_need(cfg, gamma) if uses_Q else None
                for tau_B in tau_B_grid:
                    B = benefit_vector_from_obj(benefit_obj, tau_B) if uses_B else None
                    for alpha_max in alpha_maxes:
                        S = None
                        if uses_sample_B:
                            S = compute_sample_text_benefit(p_t, B)
                            alpha = float(alpha_max) * S
                            if uses_G:
                                alpha = alpha * G.to(va_v.device)[:, None]
                            p_fused = fuse_probabilities_with_sample_gate(
                                p_v, p_t, alpha)
                            scores = p_fused.clamp_min(1e-12).log()
                        else:
                            alpha = build_prob_alpha(
                                mode, n_val, C, alpha_max, gate=G, tail_need=Q,
                                class_rel=class_rel if uses_R else None,
                                benefit=B, device=va_v.device)
                            p_fused = fuse_probabilities_with_class_gate(
                                p_v, p_t, alpha)
                            scores = p_fused.clamp_min(1e-8).log()
                        res = metricsmod.evaluate_logits(cfg, scores, labels)
                        summaries = prob_summary_payload(B, S, G, alpha)
                        rec = {
                            "Tv": Tv,
                            "Tt": Tt,
                            "alpha_max": alpha_max,
                            "eta": eta if uses_G else None,
                            "gamma": gamma if uses_Q else None,
                            "tau_B": tau_B if uses_B else None,
                            "benefit_kappa": (benefit_obj.get("benefit_kappa")
                                              if benefit_obj is not None else None),
                            "benefit_file": benefit_file,
                            "benefit_application": ("sample_expectation"
                                                    if uses_sample_B else
                                                    ("class_channel" if uses_B else None)),
                            "sample_confidence_formula": ("sum_c p_text_ic * B_c"
                                                          if uses_sample_B else None),
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "uncertainty_q_low": q_low if uses_G else None,
                            "uncertainty_q_high": q_high if uses_G else None,
                            **metric_record(res),
                            **alpha_stats(alpha),
                            "visual_val_group_avg_acc": visual_prob_rec["group_avg_acc"],
                            "visual_val_auroc": visual_prob_rec["auroc"],
                            "visual_val_auprc": visual_prob_rec["auprc"],
                            "visual_logit_val_group_avg_acc": visual_logit_rec["group_avg_acc"],
                            "visual_logit_val_auroc": visual_logit_rec["auroc"],
                            "visual_logit_val_auprc": visual_logit_rec["auprc"],
                            "visual_prob_val_group_avg_acc": visual_prob_rec["group_avg_acc"],
                            "visual_prob_val_auroc": visual_prob_rec["auroc"],
                            "visual_prob_val_auprc": visual_prob_rec["auprc"],
                            **summaries,
                        }
                        add_selection_constraint_fields(
                            rec, visual_prob_rec, args.selection_auroc_drop,
                            args.selection_auprc_drop, "probability")
                        if float(alpha_max) == 0.0:
                            assert_prob_zero_alpha_consistency(
                                p_fused, p_v, rec, visual_prob_rec)
                        search.append(rec)
                        status = "ok" if rec["passes_constraints"] else "drop"
                        print(f"  Tv={Tv:g} amax={alpha_max:g} "
                              f"eta={eta if uses_G else '-'} "
                              f"gamma={gamma if uses_Q else '-'} "
                              f"tau_B={tau_B if uses_B else '-'} "
                              f"[{status}]: " + metricsmod.format_summary(res, "val"))
                        key = selection_key(rec)
                        selected = {
                            "Tv": Tv,
                            "Tt": Tt,
                            "alpha_max": alpha_max,
                            "eta": eta if uses_G else 1.0,
                            "gamma": gamma if uses_Q else 0.0,
                            "tau_B": tau_B if uses_B else None,
                            "benefit_kappa": (benefit_obj.get("benefit_kappa")
                                              if benefit_obj is not None else None),
                            "benefit_file": rec["benefit_file"],
                            "benefit_application": rec["benefit_application"],
                            "sample_confidence_formula": rec["sample_confidence_formula"],
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "uncertainty_q_low": q_low if uses_G else None,
                            "uncertainty_q_high": q_high if uses_G else None,
                            "selection_auroc_drop": args.selection_auroc_drop,
                            "selection_auprc_drop": args.selection_auprc_drop,
                            "selection_baseline_domain": rec["selection_baseline_domain"],
                            "selection_baseline_auroc": rec["selection_baseline_auroc"],
                            "selection_baseline_auprc": rec["selection_baseline_auprc"],
                            "selection_auroc_threshold": rec["selection_auroc_threshold"],
                            "selection_auprc_threshold": rec["selection_auprc_threshold"],
                        }
                        if best is None or key > best[0]:
                            best = (key, selected, res, alpha, U, G, Q, B, S,
                                    scores, rec, p_t)

    _, selected, va_res, va_alpha, va_U, va_G, va_Q, va_B, va_S, va_scores, best_rec, va_p_t = best
    if not best_rec["passes_constraints"]:
        print("[fusion][warn] no probability-fusion candidate passed AUROC/AUPRC "
              "constraints; selected the best fallback by the same ranking.")
    drop_count = sum(1 for rec in search if not rec["passes_constraints"])
    print(f"[fusion] selected {selected} | " +
          metricsmod.format_summary(va_res, "val(best)"))
    for name in ("B_summary", "S_summary", "G_summary", "alpha_summary"):
        if name in best_rec:
            print(f"[fusion] {name}={best_rec[name]}")

    out = utils.ensure_dir(utils.method_dir(
        cfg.general.dataset_name, f"VT_{mode}_{scheme}_seed{seed}"))
    utils.save_json(os.path.join(out, "selected_fusion.json"), {
        "mode": mode,
        "fusion_domain": "prob",
        "text_scheme": scheme,
        "seed": seed,
        "visual_run_dir": visual_run_dir,
        "Tv": selected["Tv"],
        "Tt": selected["Tt"],
        "Tt_source": Tt_src,
        "logit_scale": logit_scale,
        **selected,
        "selected": selected,
        "search": search,
        "val_group_avg_acc": metricsmod.group_avg_acc(va_res),
        "alpha_stats": alpha_stats(va_alpha),
        **{k: best_rec[k] for k in ("B_summary", "S_summary", "G_summary",
                                    "alpha_summary") if k in best_rec},
        "passes_constraints": best_rec["passes_constraints"],
        "selection_drop_count": drop_count,
        "selection_total_count": len(search),
        "selection_constraints": {
            "auroc_drop": args.selection_auroc_drop,
            "auprc_drop": args.selection_auprc_drop,
        },
        "selection_note": "Candidates failing AUROC/AUPRC drop constraints are ranked after valid candidates.",
    })
    utils.save_json(os.path.join(out, "val_results.json"), {
        "mode": mode,
        "fusion_domain": "prob",
        "text_scheme": scheme,
        **selected,
        "val": metricsmod.results_to_jsonable(va_res),
        "alpha_stats": alpha_stats(va_alpha),
        **{k: best_rec[k] for k in ("B_summary", "S_summary", "G_summary",
                                    "alpha_summary") if k in best_rec},
        "passes_constraints": best_rec["passes_constraints"],
    })
    utils.save_json(os.path.join(out, "per_class_val.json"), {
        **selected,
        "per_class_acc": list(va_res["class_acc"]),
    })
    sample_rel = (calibration.sample_text_reliability(va_t, class_rel, Tt)
                  if class_rel is not None else None)
    va_ent, va_margin = uncertainty_components(va_v, selected["Tv"])
    save_diagnostics(os.path.join(out, "val_diagnostics.csv"),
                     labels, va_v, va_t, va_scores, va_U, sample_rel, va_alpha,
                     gate=va_G, tail_need=va_Q, class_rel=class_rel,
                     benefit=va_B, visual_entropy=va_ent,
                     visual_margin=va_margin,
                     sample_text_confidence=va_S, text_probs=va_p_t)
    if benefit_obj is not None and va_B is not None:
        save_benefit_class_diagnostics(
            os.path.join(out, "val_class_diagnostics.csv"),
            cfg, benefit_obj, va_B, labels, va_v, va_scores, "val")
    return selected, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default=None, choices=list(ALL_MODES))
    ap.add_argument("--fusion-domain", default=None, choices=["logit", "prob"],
                    help="optional convenience: --fusion-domain prob --mode fixed -> prob_fixed")
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
    ap.add_argument("--alpha-maxes", default=None,
                    help="comma list for probability-domain alpha_max search")
    ap.add_argument("--gammas", default=None,
                    help="comma list for tail-need Q_c strength in probability modes")
    ap.add_argument("--etas", default=None,
                    help="comma list for U = eta*entropy + (1-eta)*margin uncertainty")
    ap.add_argument("--tau-Bs", "--tau-bs", dest="tau_Bs", default=None,
                    help="comma list for mapping conservative text gains to B_c")
    ap.add_argument("--benefit-file", default=None,
                    help="text_benefit_<scheme>.json; defaults to visual run dir")
    ap.add_argument("--uncertainty-quantiles", default="0.3,0.8",
                    help="low,high validation quantiles for mapping uncertainty to G")
    ap.add_argument("--selection-auroc-drop", type=float, default=1.0,
                    help="prob modes: max allowed validation AUROC drop vs visual baseline")
    ap.add_argument("--selection-auprc-drop", type=float, default=1.0,
                    help="prob modes: max allowed validation AUPRC drop vs visual baseline")
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
    if args.fusion_domain == "prob" and mode in calibration.MODES:
        mode = f"prob_{mode}" if mode == "fixed" else mode
        if mode not in PROB_MODES:
            raise SystemExit("--fusion-domain prob currently supports --mode fixed via "
                             "prob_fixed; use explicit prob_G/prob_Q/... for gated modes")
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
    benefit_file = args.benefit_file or (selected or {}).get("benefit_file")
    benefit_obj = get_text_benefit(
        cfg, scheme, visual_run_dir, ckpt, mode, benefit_file)

    if args.eval_split == "test":
        if loaded is None:
            raise SystemExit("--eval-split test requires --load-best-config; "
                             "test mode does not search hyper-parameters")
        out = utils.ensure_dir(os.path.dirname(args.load_best_config))
        run_eval_split(cfg, "test", mode, scheme, selected, head, protos,
                       logit_scale, meta, class_rel, alpha_max, out, benefit_obj)
        print(f"[fusion] outputs -> {out}")
        return

    selected, out = validation_search(
        cfg, args, mode, scheme, head, protos, logit_scale, meta, class_rel,
        Tt, Tt_src, seed, visual_run_dir, alpha_max, benefit_obj)

    if args.test:
        run_eval_split(cfg, "test", mode, scheme, selected, head, protos,
                       logit_scale, meta, class_rel, alpha_max, out, benefit_obj)
    print(f"[fusion] outputs -> {out}")


if __name__ == "__main__":
    main()
