"""Stage 2 - train the visual linear head on frozen cached features ('V' baseline).

- Loads Stage-1 train/val features+labels (test is NOT loaded here).
- Trains ONLY a linear head; BiomedCLIP features stay frozen (precomputed).
- Default plain CrossEntropy.
- Selects the best checkpoint by VALIDATION group-average accuracy (= MONICA's
  best.pt criterion). Test is evaluated separately, once, via evaluate_test.py.
- Saves best.pt, per-epoch logs, a config snapshot (incl. seed), and per-class
  validation results.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.train --config biomedclip_ltc/configs/isic_100.yml \
        --seed 1 --lr 1e-3 --batch-size 512 --epochs 100 --weight-decay 0.0
"""
import argparse
import os

import torch

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import utils
from biomedclip_ltc.losses import get_loss
from biomedclip_ltc.model import VisualHead


# short loss tags so visual-only runs are NOT mislabeled with a text scheme
LOSS_ABBR = {"CE": "CE", "BalancedSoftmax": "BS", "LogitAdjust": "LA"}


def build_run_name(cfg, variant):
    bm = cfgmod.bm
    loss = LOSS_ABBR.get(bm(cfg, "lt_loss"), bm(cfg, "lt_loss"))
    return (f"{variant}_{loss}_seed{cfg.general.seed}"
            f"_lr{bm(cfg, 'lr')}_bs{bm(cfg, 'batch_size')}_ep{bm(cfg, 'epochs')}")


def run_dir(cfg, variant):
    return os.path.join("outputs", "biomedclip_ltc", cfg.general.dataset_name,
                        build_run_name(cfg, variant))


def apply_overrides(cfg, args):
    if args.seed is not None:
        cfg.general.seed = args.seed
    for k_cli, k_cfg in [("lr", "lr"), ("batch_size", "batch_size"),
                         ("epochs", "epochs"), ("weight_decay", "weight_decay"),
                         ("lt_loss", "lt_loss"), ("text_scheme", "text_scheme")]:
        v = getattr(args, k_cli)
        if v is not None:
            cfg.biomedclip[k_cfg] = v


def evaluate_split(cfg, model, feats, labels, device, batch_size):
    """Forward a whole split and return (logits_cpu, MONICA metric dict)."""
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):
            xb = feats[i:i + batch_size].to(device)
            logits.append(model(xb).cpu())
    logits = torch.cat(logits, dim=0)
    return logits, metricsmod.evaluate_logits(cfg, logits, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="V", choices=["V"],
                    help="Stage 2 implements the visual baseline 'V' only")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--lt-loss", dest="lt_loss", default=None,
                    choices=["CE", "BalancedSoftmax", "LogitAdjust"])
    ap.add_argument("--text-scheme", dest="text_scheme", default=None,
                    choices=["P0", "P1", "P2"])
    ap.add_argument("--cos-lr", action="store_true", help="cosine LR schedule")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    apply_overrides(cfg, args)
    seed = int(cfg.general.seed)
    lr = float(cfgmod.bm(cfg, "lr"))
    bs = int(cfgmod.bm(cfg, "batch_size"))
    epochs = int(cfgmod.bm(cfg, "epochs"))
    wd = float(cfgmod.bm(cfg, "weight_decay"))
    C = cfg.general.num_classes
    D = int(cfgmod.bm(cfg, "feature_dim"))

    utils.set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = utils.ensure_dir(run_dir(cfg, args.variant))
    log_path = os.path.join(out, "logs.txt")
    print(f"[Stage2] variant={args.variant} dataset={cfg.general.dataset_name} "
          f"seed={seed} lr={lr} bs={bs} epochs={epochs} wd={wd} device={device}")
    print(f"[Stage2] output dir: {out}")

    # --- load + integrity check (train/val only; NEVER test) ---
    meta = datamod.load_extract_meta(cfg)
    tr_feats, tr_labels = datamod.load_split(cfg, "train")
    va_feats, va_labels = datamod.load_split(cfg, "val")
    rep_tr = datamod.check_cache_integrity(cfg, tr_feats, tr_labels, "train", meta)
    rep_va = datamod.check_cache_integrity(cfg, va_feats, va_labels, "val", meta)
    print("[Stage2] cache integrity OK:", rep_tr, rep_va)

    cls_num_list = datamod.get_cls_num_list(tr_labels, C)
    model = VisualHead(D, C).to(device)
    criterion = get_loss(cfg, cls_num_list)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_cos = args.cos_lr or bool(cfgmod.bm(cfg, "cos_lr"))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0)
                 if use_cos else None)

    # config snapshot (with seed + cls counts)
    utils.save_json(os.path.join(out, "config_snapshot.json"), {
        "resolved_config": cfg.__getstate__(),
        "seed": seed, "variant": args.variant,
        "lr": lr, "batch_size": bs, "epochs": epochs, "weight_decay": wd,
        "lt_loss": cfgmod.bm(cfg, "lt_loss"),
        "cls_num_list_train": cls_num_list,
        "integrity": {"train": rep_tr, "val": rep_va},
    })

    N = tr_feats.shape[0]
    best_acc = -1.0
    best_epoch = -1
    with open(log_path, "w") as logf:
        logf.write(f"# Stage2 visual baseline | seed={seed} lr={lr} bs={bs} "
                   f"epochs={epochs} wd={wd} loss={cfgmod.bm(cfg, 'lt_loss')}\n")
        for epoch in range(epochs):
            model.train()
            g = torch.Generator().manual_seed(seed * 100003 + epoch)
            perm = torch.randperm(N, generator=g)
            running = 0.0
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                xb = tr_feats[idx].to(device)
                yb = tr_labels[idx].to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.shape[0]
            if scheduler is not None:
                scheduler.step()
            train_loss = running / N

            _, va_res = evaluate_split(cfg, model, va_feats, va_labels, device, bs)
            va_acc = metricsmod.group_avg_acc(va_res)
            line = (f"epoch {epoch} | train_loss {train_loss:.4f} | "
                    + metricsmod.format_summary(va_res, "val"))
            print(line)
            logf.write(line + "\n")
            logf.write("  val class acc: "
                       + " ".join(f"{a:.2f}" for a in va_res["class_acc"]) + "\n")
            logf.flush()

            if va_acc > best_acc:
                best_acc, best_epoch = va_acc, epoch
                torch.save({
                    "head_state": model.state_dict(),
                    "feature_dim": D, "num_classes": C, "variant": args.variant,
                    "seed": seed, "best_epoch": epoch,
                    "best_val_group_acc": best_acc,
                    "cfg_snapshot": cfg.__getstate__(),
                }, os.path.join(out, "best.pt"))
                utils.save_json(os.path.join(out, "per_class_val_best.json"), {
                    "best_epoch": epoch, "best_val_group_acc": best_acc,
                    "val": metricsmod.results_to_jsonable(va_res),
                })
                logf.write(f"  * new best val groupAvgAcc={best_acc:.4f} "
                           f"@ epoch {epoch}\n")
                logf.flush()

        logf.write(f"# best val groupAvgAcc={best_acc:.4f} @ epoch {best_epoch}\n")
    print(f"[Stage2] done. best val groupAvgAcc={best_acc:.4f} @ epoch {best_epoch}")
    print(f"[Stage2] run `python -m biomedclip_ltc.evaluate_test --run-dir {out}` "
          "for the single final test evaluation.")


if __name__ == "__main__":
    main()
