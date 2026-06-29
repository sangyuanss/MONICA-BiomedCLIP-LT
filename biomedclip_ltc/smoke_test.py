"""Stage 2 smoke test — runs the full train -> evaluate_test path on a tiny
SYNTHETIC cache (no BiomedCLIP, no images, no real splits required).

Generates class-separable L2-normalized features for a 6-class imbalanced toy
dataset, writes them in the Stage-1 cache layout, then trains the visual head and
runs the single final test evaluation, asserting the pipeline completes and beats
chance. Useful to validate the code on CPU before renting a GPU.

Usage (from the MONICA repo root):
    python -m biomedclip_ltc.smoke_test
"""
import glob
import os
import shutil
import sys
import tempfile

import numpy as np

from biomedclip_ltc import train, evaluate_test, utils

C, D, IR = 6, 64, 999
HEAD, MEDIUM = 2, 4
TRAIN_PER = [200, 150, 100, 60, 40, 20]
EVAL_PER = 20


def _gen(rng, centers, n_per_class):
    feats, labels = [], []
    for c in range(C):
        feats.append(centers[c] + 0.6 * rng.normal(size=(n_per_class[c], D)))
        labels += [c] * n_per_class[c]
    X = np.concatenate(feats, 0)
    y = np.array(labels, dtype=np.int64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    p = rng.permutation(len(y))
    return X[p].astype(np.float32), y[p]


def main():
    tmp = tempfile.mkdtemp(prefix="biomedclip_smoke_")
    feat_root = os.path.join(tmp, "features")
    ds_dir = utils.ensure_dir(os.path.join(feat_root, "smoke"))
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(C, D))

    sizes = {}
    for split, npc in [("train", TRAIN_PER),
                       ("val", [EVAL_PER] * C), ("test", [EVAL_PER] * C)]:
        X, y = _gen(rng, centers, npc)
        np.save(os.path.join(ds_dir, f"{split}_{IR}_feats.npy"), X)
        np.save(os.path.join(ds_dir, f"{split}_{IR}_labels.npy"), y)
        sizes[split] = len(y)
    utils.save_json(os.path.join(ds_dir, "extract_meta.json"), {
        "hub_id": "smoke", "feature_dim": D, "normalization": "l2 (synthetic)",
        "split_sizes": sizes, "prompt_version": "smoke"})

    yml = os.path.join(tmp, "smoke.yml")
    with open(yml, "w") as f:
        f.write(f"""general:
  seed: 1
  num_classes: {C}
  dataset_name: 'smoke'
datasets:
  imbalance_ratio: {IR}
  head: {HEAD}
  medium: {MEDIUM}
  tail: {C}
biomedclip:
  hub_id: 'smoke'
  feature_dim: {D}
  feature_root: '{feat_root.replace(os.sep, '/')}'
  proto_root: '{feat_root.replace(os.sep, '/')}'
  lr: 0.01
  batch_size: 64
  epochs: 8
  weight_decay: 0.0
  lt_loss: 'CE'
  text_scheme: 'P1'
""")

    run_glob = os.path.join("outputs", "biomedclip_ltc", "smoke", "*")
    for d in glob.glob(run_glob):
        shutil.rmtree(d, ignore_errors=True)

    try:
        sys.argv = ["train", "--config", yml]
        train.main()
        run_dirs = glob.glob(run_glob)
        assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"
        run_dir = run_dirs[0]
        assert os.path.exists(os.path.join(run_dir, "best.pt"))
        assert os.path.exists(os.path.join(run_dir, "config_snapshot.json"))
        assert os.path.exists(os.path.join(run_dir, "per_class_val_best.json"))

        sys.argv = ["evaluate_test", "--run-dir", run_dir]
        evaluate_test.main()
        res = utils.load_json(os.path.join(run_dir, "test_results.json"))
        avg = res["test"]["accuracy"][3]
        assert avg > 100.0 / C, f"test groupAvgAcc {avg:.2f} not above chance"
        print(f"\nSMOKE TEST PASSED — test groupAvgAcc={avg:.2f} (chance≈{100.0/C:.1f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
