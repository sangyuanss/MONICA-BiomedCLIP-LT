"""Train-only OOF text benefit estimation for probability-domain fusion.

This module estimates a per-class text benefit signal B without using validation
or test labels. For each training sample, a K-fold out-of-fold visual head
prediction is compared against the frozen text branch:

    gain_i = CE_visual_oof_i - CE_text_i
           = log p_text(y_i) - log p_visual_oof(y_i)

Positive gain means the text branch is better than the visual branch for that
training sample. Per-class conservative gains are saved; the fusion stage later
searches tau_B and maps gains to B_c in [0, 1].
"""
import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import utils
from biomedclip_ltc.losses import get_loss
from biomedclip_ltc.model import VisualHead

BENEFIT_DEFINITION = "oof_ce_text_minus_visual_v1"


def load_text_temperature(cfg, scheme, override=None):
    if override is not None:
        return float(override), "arg"
    sel = os.path.join(utils.method_dir(cfg.general.dataset_name, f"T_{scheme}"),
                       "selected_temperature.json")
    if os.path.exists(sel):
        return float(utils.load_json(sel)["selected_temperature"]), sel
    return 1.0, "default(1.0; run evaluate_text first for tuned Tt)"


def _cfg_snapshot_from_run(visual_run_dir, ckpt):
    snap = ckpt.get("cfg_snapshot")
    if snap:
        return snap, "best.pt:cfg_snapshot"
    path = os.path.join(visual_run_dir, "config_snapshot.json")
    if os.path.exists(path):
        obj = utils.load_json(path)
        return obj.get("resolved_config") or obj, path
    return {}, "missing"


def _nested_get(obj, path, default=None):
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def recover_training_params(cfg, visual_run_dir, ckpt, lt_loss_fallback=None):
    """Recover OOF head training params from visual-run metadata, never path names."""
    snap, source = _cfg_snapshot_from_run(visual_run_dir, ckpt)
    b = snap.get("biomedclip", {}) if isinstance(snap, dict) else {}
    g = snap.get("general", {}) if isinstance(snap, dict) else {}

    loss = b.get("lt_loss")
    if loss is None:
        loss = _nested_get(snap, ["resolved_config", "biomedclip", "lt_loss"])
    if loss is None:
        if lt_loss_fallback is None:
            raise SystemExit(
                "Could not recover visual lt_loss from best.pt/config_snapshot.json. "
                "Pass --lt-loss explicitly; refusing to guess from the directory name.")
        loss = lt_loss_fallback

    params = {
        "seed": int(ckpt.get("seed", g.get("seed", cfg.general.seed))),
        "lr": float(b.get("lr", cfgmod.bm(cfg, "lr"))),
        "batch_size": int(b.get("batch_size", cfgmod.bm(cfg, "batch_size"))),
        "epochs": int(b.get("epochs", cfgmod.bm(cfg, "epochs"))),
        "weight_decay": float(b.get("weight_decay", cfgmod.bm(cfg, "weight_decay"))),
        "cos_lr": bool(b.get("cos_lr", cfgmod.bm(cfg, "cos_lr"))),
        "lt_loss": loss,
        "metadata_source": source,
    }
    return params


def benefit_path(visual_run_dir, scheme):
    return os.path.join(visual_run_dir, f"text_benefit_{scheme}.json")


def benefit_sample_path(visual_run_dir, scheme):
    return os.path.join(visual_run_dir, f"text_benefit_{scheme}_per_sample.csv")


def train_fold_head(cfg, train_feats, train_labels, params, device, fold_seed):
    C = cfg.general.num_classes
    D = int(cfgmod.bm(cfg, "feature_dim"))
    model = VisualHead(D, C).to(device)

    cfg.biomedclip["lt_loss"] = params["lt_loss"]
    cls_num_list = datamod.get_cls_num_list(train_labels.cpu(), C)
    criterion = get_loss(cfg, cls_num_list)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params["epochs"], eta_min=0.0) if params["cos_lr"] else None)

    N = train_feats.shape[0]
    bs = params["batch_size"]
    for epoch in range(params["epochs"]):
        model.train()
        g = torch.Generator().manual_seed(fold_seed * 100003 + epoch)
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = train_feats[idx].to(device)
            yb = train_labels[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model


@torch.no_grad()
def forward_logits(model, feats, device, batch_size):
    model.eval()
    out = []
    for i in range(0, feats.shape[0], batch_size):
        out.append(model(feats[i:i + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


def compute_oof_visual_logits(cfg, feats, labels, params, folds, seed, device):
    counts = torch.bincount(labels, minlength=cfg.general.num_classes)
    min_class_count = int(counts[counts > 0].min().item())
    effective_folds = min(int(folds), min_class_count)
    if effective_folds < 2:
        raise SystemExit(
            f"effective_folds={effective_folds} < 2; refusing to use in-sample predictions.")

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    y_np = labels.numpy()
    oof_logits = torch.empty(feats.shape[0], cfg.general.num_classes)
    pred_count = torch.zeros(feats.shape[0], dtype=torch.long)
    fold_ids = torch.full((feats.shape[0],), -1, dtype=torch.long)

    for fold, (tr_idx_np, va_idx_np) in enumerate(skf.split(np.zeros(len(y_np)), y_np)):
        tr_idx = torch.from_numpy(tr_idx_np).long()
        va_idx = torch.from_numpy(va_idx_np).long()
        if set(tr_idx_np).intersection(set(va_idx_np)):
            raise RuntimeError(f"fold {fold}: train/OOF indices overlap")
        print(f"[benefit] fold {fold + 1}/{effective_folds}: "
              f"train={len(tr_idx)} oof={len(va_idx)}")
        model = train_fold_head(
            cfg, feats[tr_idx], labels[tr_idx], params, device, seed + fold)
        oof_logits[va_idx] = forward_logits(
            model, feats[va_idx], device, params["batch_size"])
        pred_count[va_idx] += 1
        fold_ids[va_idx] = fold

    if not torch.equal(pred_count, torch.ones_like(pred_count)):
        bad = torch.where(pred_count != 1)[0].tolist()[:10]
        raise RuntimeError(f"OOF prediction count is not exactly one for samples {bad}")
    if (fold_ids < 0).any():
        raise RuntimeError("OOF fold assignment missing for some samples")
    return oof_logits, fold_ids, effective_folds


def classwise_stats(values, labels, num_classes):
    means, stds, ses = [], [], []
    for c in range(num_classes):
        vals = values[labels == c]
        if vals.numel() == 0:
            means.append(float("nan"))
            stds.append(float("nan"))
            ses.append(float("nan"))
            continue
        mean = float(vals.mean().item())
        std = float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0
        se = std / math.sqrt(int(vals.numel()))
        means.append(mean)
        stds.append(std)
        ses.append(se)
    return means, stds, ses


def accuracy_by_class(pred, labels, num_classes):
    out = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            out.append(float((pred[mask] == labels[mask]).float().mean().item() * 100.0))
        else:
            out.append(0.0)
    return out


def benefit_from_conservative_gain(conservative_gain, tau_B):
    gains = torch.as_tensor(conservative_gain, dtype=torch.float)
    B = 2.0 * torch.sigmoid(gains / float(tau_B)) - 1.0
    return B.clamp(0.0, 1.0)


def load_benefit(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing text benefit file {path}. Run `python -m biomedclip_ltc.text_benefit` first.")
    obj = utils.load_json(path)
    if obj.get("benefit_definition") != BENEFIT_DEFINITION:
        raise ValueError(f"{path} has incompatible benefit_definition={obj.get('benefit_definition')}")
    return obj


def validate_benefit(obj, cfg, visual_run_dir, text_scheme, ckpt=None):
    errors = []
    if obj.get("dataset") != cfg.general.dataset_name:
        errors.append(f"dataset {obj.get('dataset')} != {cfg.general.dataset_name}")
    if int(obj.get("num_classes", -1)) != int(cfg.general.num_classes):
        errors.append(f"num_classes {obj.get('num_classes')} != {cfg.general.num_classes}")
    if obj.get("text_scheme") != text_scheme:
        errors.append(f"text_scheme {obj.get('text_scheme')} != {text_scheme}")
    saved_run = os.path.abspath(obj.get("visual_run_dir", ""))
    cur_run = os.path.abspath(visual_run_dir)
    if saved_run != cur_run:
        errors.append(f"visual_run_dir {saved_run} != {cur_run}")
    if ckpt is not None and "seed" in obj and int(obj["seed"]) != int(ckpt.get("seed", obj["seed"])):
        errors.append(f"seed {obj['seed']} != checkpoint seed {ckpt.get('seed')}")
    if ckpt is not None:
        params = recover_training_params(cfg, visual_run_dir, ckpt, obj.get("visual_loss"))
        if obj.get("visual_loss") != params["lt_loss"]:
            errors.append(f"visual_loss {obj.get('visual_loss')} != run loss {params['lt_loss']}")
    if errors:
        raise ValueError("Benefit file mismatch: " + "; ".join(errors))


def save_per_sample_csv(path, labels, fold_ids, p_visual_true, p_text_true,
                        ce_visual, ce_text, gain, visual_pred, text_pred):
    utils.ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_index", "label", "fold", "visual_true_prob_oof",
            "text_true_prob", "visual_ce_oof", "text_ce", "gain",
            "visual_pred_oof", "text_pred", "visual_correct_oof", "text_correct",
        ])
        writer.writeheader()
        for i in range(labels.shape[0]):
            writer.writerow({
                "sample_index": i,
                "label": int(labels[i]),
                "fold": int(fold_ids[i]),
                "visual_true_prob_oof": float(p_visual_true[i]),
                "text_true_prob": float(p_text_true[i]),
                "visual_ce_oof": float(ce_visual[i]),
                "text_ce": float(ce_text[i]),
                "gain": float(gain[i]),
                "visual_pred_oof": int(visual_pred[i]),
                "text_pred": int(text_pred[i]),
                "visual_correct_oof": int(visual_pred[i] == labels[i]),
                "text_correct": int(text_pred[i] == labels[i]),
            })


def compute_and_save(cfg, visual_run_dir, scheme, folds, benefit_kappa, seed=None,
                     device=None, output=None, lt_loss=None, overwrite=False,
                     visual_temperature=None, text_temperature=None):
    visual_run_dir = os.path.abspath(visual_run_dir)
    out_json = output or benefit_path(visual_run_dir, scheme)
    out_csv = out_json.replace(".json", "_per_sample.csv")
    if os.path.exists(out_json) and not overwrite:
        raise SystemExit(f"{out_json} exists; pass --overwrite to recompute.")

    ckpt_path = os.path.join(visual_run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = recover_training_params(cfg, visual_run_dir, ckpt, lt_loss)
    if seed is None:
        seed = int(params["seed"])
    Tv = float(visual_temperature if visual_temperature is not None
               else cfgmod.bm(cfg, "visual_temperature"))
    Tt, Tt_src = load_text_temperature(cfg, scheme, text_temperature)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    feats, labels = datamod.load_split(cfg, "train")
    datamod.check_cache_integrity(cfg, feats, labels, "train",
                                  datamod.load_extract_meta(cfg))
    oof_logits, fold_ids, effective_folds = compute_oof_visual_logits(
        cfg, feats, labels, params, folds, int(seed), device)

    protos = datamod.load_text_prototypes(cfg, scheme)
    logit_scale = datamod.get_logit_scale(cfg)
    text_logits = logit_scale * (feats @ protos.t())

    p_visual = F.softmax(oof_logits / Tv, dim=1)
    p_text = F.softmax(text_logits / Tt, dim=1)
    p_visual_true = p_visual.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8)
    p_text_true = p_text.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8)
    ce_visual = -p_visual_true.log()
    ce_text = -p_text_true.log()
    gain = ce_visual - ce_text
    visual_pred = p_visual.argmax(dim=1)
    text_pred = p_text.argmax(dim=1)

    C = cfg.general.num_classes
    counts = torch.bincount(labels, minlength=C).float()
    mean_gain, std_gain, se_gain = classwise_stats(gain, labels, C)
    conservative_gain = [
        mean_gain[c] - float(benefit_kappa) * se_gain[c] for c in range(C)
    ]

    cls_visual_acc = accuracy_by_class(visual_pred, labels, C)
    cls_text_acc = accuracy_by_class(text_pred, labels, C)
    overall_visual_acc = float((visual_pred == labels).float().mean().item() * 100.0)
    overall_text_acc = float((text_pred == labels).float().mean().item() * 100.0)

    metadata = {
        "benefit_definition": BENEFIT_DEFINITION,
        "dataset": cfg.general.dataset_name,
        "num_classes": C,
        "visual_run_dir": visual_run_dir,
        "visual_loss": params["lt_loss"],
        "seed": int(seed),
        "text_scheme": scheme,
        "folds_requested": int(folds),
        "folds_effective": int(effective_folds),
        "benefit_kappa": float(benefit_kappa),
        "visual_temperature": Tv,
        "text_temperature": Tt,
        "text_temperature_source": Tt_src,
        "logit_scale": logit_scale,
        "training_params": params,
        "class_counts": counts.tolist(),
        "class_mean_gain": mean_gain,
        "class_std_gain": std_gain,
        "class_standard_error": se_gain,
        "class_conservative_gain": conservative_gain,
        "class_visual_oof_accuracy": cls_visual_acc,
        "class_text_accuracy": cls_text_acc,
        "overall_visual_oof_accuracy": overall_visual_acc,
        "overall_text_accuracy": overall_text_acc,
    }
    utils.save_json(out_json, metadata)
    save_per_sample_csv(out_csv, labels, fold_ids, p_visual_true, p_text_true,
                        ce_visual, ce_text, gain, visual_pred, text_pred)
    print(f"[benefit] saved -> {out_json}")
    print(f"[benefit] per-sample -> {out_csv}")
    print("[benefit] conservative_gain:",
          " ".join(f"c{c}={conservative_gain[c]:.3f}" for c in range(C)))
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--visual-run-dir", required=True)
    ap.add_argument("--text-scheme", required=True, choices=["P0", "P1", "P2"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--benefit-kappa", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--lt-loss", dest="lt_loss", default=None,
                    choices=["CE", "BalancedSoftmax", "LogitAdjust"],
                    help="explicit fallback only if visual-run metadata lacks lt_loss")
    ap.add_argument("--visual-temperature", type=float, default=None)
    ap.add_argument("--text-temperature", type=float, default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    compute_and_save(
        cfg, args.visual_run_dir, args.text_scheme, args.folds,
        args.benefit_kappa, seed=args.seed, device=args.device,
        output=args.output, lt_loss=args.lt_loss, overwrite=args.overwrite,
        visual_temperature=args.visual_temperature,
        text_temperature=args.text_temperature)


if __name__ == "__main__":
    main()
