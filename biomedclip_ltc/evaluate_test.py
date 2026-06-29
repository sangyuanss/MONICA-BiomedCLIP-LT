"""Stage 2 - SINGLE final test evaluation.

Run ONLY after training has fixed the best (val-selected) checkpoint. Loads
best.pt from a run directory, rebuilds the config from its snapshot, evaluates
the TEST split exactly once, and writes test_results.json. Test is never used
for any selection.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.evaluate_test --run-dir outputs/biomedclip_ltc/isic/<run_name>
"""
import argparse
import os

import torch

from utils.setup_configs import Config
from biomedclip_ltc import data as datamod
from biomedclip_ltc import metrics as metricsmod
from biomedclip_ltc import utils
from biomedclip_ltc.model import VisualHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    ckpt_path = os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best.pt in {args.run_dir}; train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(ckpt["cfg_snapshot"])
    D, C = ckpt["feature_dim"], ckpt["num_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] run={args.run_dir} variant={ckpt['variant']} "
          f"best_epoch={ckpt['best_epoch']} "
          f"val_groupAvgAcc={ckpt['best_val_group_acc']:.4f} device={device}")

    model = VisualHead(D, C).to(device)
    model.load_state_dict(ckpt["head_state"])
    model.eval()

    meta = datamod.load_extract_meta(cfg)
    te_feats, te_labels = datamod.load_split(cfg, "test")
    rep = datamod.check_cache_integrity(cfg, te_feats, te_labels, "test", meta)
    print("[test] cache integrity OK:", rep)

    logits = []
    with torch.no_grad():
        for i in range(0, te_feats.shape[0], args.batch_size):
            logits.append(model(te_feats[i:i + args.batch_size].to(device)).cpu())
    logits = torch.cat(logits, dim=0)
    res = metricsmod.evaluate_logits(cfg, logits, te_labels)

    print(metricsmod.format_summary(res, "TEST"))
    h, m, t, avg = res["accuracy"]
    print(f"[test] overall(groupAvg)={avg:.2f}  Head={h:.2f}  Medium={m:.2f}  Tail={t:.2f}")
    print("[test] per-class acc:", " ".join(f"{a:.2f}" for a in res["class_acc"]))

    out_json = os.path.join(args.run_dir, "test_results.json")
    utils.save_json(out_json, {
        "run_dir": args.run_dir,
        "variant": ckpt["variant"],
        "best_epoch": ckpt["best_epoch"],
        "val_group_avg_acc": ckpt["best_val_group_acc"],
        "integrity": rep,
        "test": metricsmod.results_to_jsonable(res),
    })
    with open(os.path.join(args.run_dir, "logs.txt"), "a") as f:
        f.write("# FINAL TEST (single evaluation)\n")
        f.write(metricsmod.format_summary(res, "TEST") + "\n")
        f.write("  test class acc: "
                + " ".join(f"{a:.2f}" for a in res["class_acc"]) + "\n")
    print(f"[test] wrote {out_json}")


if __name__ == "__main__":
    main()
