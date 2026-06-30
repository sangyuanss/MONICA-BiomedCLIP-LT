"""CPU smoke test for the BiomedCLIP-LT add-on.

The test builds a tiny synthetic frozen-feature cache, then runs:
train -> evaluate_test -> evaluate_text -> text_reliability -> text_benefit ->
logit fusion -> probability fusion.

It requires no BiomedCLIP checkpoint, image files, or real dataset splits.

Usage from the MONICA repo root:
    python -m biomedclip_ltc.smoke_test
"""
import csv
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from biomedclip_ltc import (calibration, evaluate_fusion, evaluate_test,
                            evaluate_text, text_benefit, text_reliability,
                            train, utils)

C, D, IR = 6, 64, 999
HEAD, MEDIUM = 2, 4
TRAIN_PER = [200, 150, 100, 60, 40, 20]
EVAL_PER = 20


def _gen(rng, centers, n_per_class):
    feats, labels = [], []
    for c in range(C):
        noise = 1.6 if c == C - 1 else 0.6
        feats.append(centers[c] + noise * rng.normal(size=(n_per_class[c], D)))
        labels += [c] * n_per_class[c]
    X = np.concatenate(feats, 0)
    y = np.array(labels, dtype=np.int64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    p = rng.permutation(len(y))
    return X[p].astype(np.float32), y[p]


def _assert_benefit_smoke(path, train_size):
    obj = utils.load_json(path)
    gains = np.array(obj["class_conservative_gain"], dtype=np.float32)
    assert np.isfinite(gains).all(), gains
    assert gains.max() > 0.0, f"expected at least one text-helpful class, got {gains}"
    assert gains.min() < 0.0, f"expected at least one visual-better class, got {gains}"

    B = text_benefit.benefit_from_conservative_gain(gains, tau_B=0.5)
    assert torch.all((B >= 0.0) & (B <= 1.0)), B
    order = torch.as_tensor(np.argsort(gains))
    assert torch.all(B[order][1:] >= B[order][:-1] - 1e-7), (gains, B)

    with open(path.replace(".json", "_per_sample.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == train_size
    assert all(int(r["fold"]) >= 0 for r in rows)
    assert all("visual_ce_oof" in r and "text_ce" in r and "gain" in r
               for r in rows)


def _assert_sample_gate_smoke():
    B = torch.tensor([0.0, 0.0, 0.2, 0.9])
    p_text_high = torch.tensor([[0.02, 0.03, 0.05, 0.90]])
    p_text_low = torch.tensor([[0.90, 0.05, 0.03, 0.02]])
    S_high = evaluate_fusion.compute_sample_text_benefit(p_text_high, B)
    S_low = evaluate_fusion.compute_sample_text_benefit(p_text_low, B)
    assert S_high.shape == (1, 1)
    assert S_low.shape == (1, 1)
    assert S_high.item() > S_low.item()

    p_v = torch.softmax(torch.randn(4, C), dim=1)
    p_t = torch.softmax(torch.randn(4, C), dim=1)
    S = evaluate_fusion.compute_sample_text_benefit(p_t, torch.rand(C))
    alpha = 0.3 * S
    assert alpha.shape == (4, 1)
    assert alpha.shape != (4, C)

    zero_alpha = torch.zeros(4, 1)
    p_fused_zero = evaluate_fusion.fuse_probabilities_with_sample_gate(
        p_v, p_t, zero_alpha)
    assert torch.allclose(p_fused_zero, p_v, atol=1e-7)

    p_fused = evaluate_fusion.fuse_probabilities_with_sample_gate(p_v, p_t, alpha)
    assert torch.allclose(p_fused.sum(dim=1), torch.ones(4), atol=1e-6)

    class_alpha = evaluate_fusion.build_prob_alpha(
        "prob_GB", 4, C, 0.2, gate=torch.ones(4), benefit=torch.ones(C),
        device=p_v.device)
    assert class_alpha.shape == (4, C)


def main():
    tmp = tempfile.mkdtemp(prefix="biomedclip_smoke_")
    feat_root = os.path.join(tmp, "features")
    ds_dir = utils.ensure_dir(os.path.join(feat_root, "smoke"))
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(C, D))

    sizes = {}
    for split, npc in [("train", TRAIN_PER),
                       ("val", [EVAL_PER] * C),
                       ("test", [EVAL_PER] * C)]:
        X, y = _gen(rng, centers, npc)
        np.save(os.path.join(ds_dir, f"{split}_{IR}_feats.npy"), X)
        np.save(os.path.join(ds_dir, f"{split}_{IR}_labels.npy"), y)
        sizes[split] = len(y)
    utils.save_json(os.path.join(ds_dir, "extract_meta.json"), {
        "hub_id": "smoke",
        "feature_dim": D,
        "normalization": "l2 (synthetic)",
        "split_sizes": sizes,
        "prompt_version": "smoke",
    })

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

    # Class 0 text is intentionally misleading; the rare last class keeps a
    # clean text anchor, giving text-benefit B both positive and negative cases.
    proto_src = centers.copy()
    proto_src[0] = centers[1]
    protos = proto_src / np.linalg.norm(proto_src, axis=1, keepdims=True)
    np.save(os.path.join(feat_root, "smoke_P1.npy"), protos.astype(np.float32))

    smoke_out = os.path.join("outputs", "biomedclip_ltc", "smoke")
    shutil.rmtree(smoke_out, ignore_errors=True)

    try:
        sys.argv = ["train", "--config", yml]
        train.main()
        run_dirs = [d for d in glob.glob(os.path.join(smoke_out, "V_*"))]
        assert len(run_dirs) == 1, f"expected 1 visual run dir, got {run_dirs}"
        run_dir = run_dirs[0]
        for f in ("best.pt", "config_snapshot.json", "per_class_val_best.json"):
            assert os.path.exists(os.path.join(run_dir, f)), f
        assert "V_CE_seed1" in os.path.basename(run_dir), \
            f"run name not renamed: {run_dir}"

        sys.argv = ["evaluate_test", "--run-dir", run_dir]
        evaluate_test.main()
        avg = utils.load_json(f"{run_dir}/test_results.json")["test"]["accuracy"][3]
        assert avg > 100.0 / C, f"visual test groupAvgAcc {avg:.2f} not above chance"

        sys.argv = ["evaluate_text", "--config", yml, "--text-scheme", "P1"]
        evaluate_text.main()
        assert os.path.exists(f"{smoke_out}/T_P1/selected_temperature.json")
        assert os.path.exists(f"{smoke_out}/T_P1/val_results.json")

        sys.argv = ["text_reliability", "--config", yml, "--text-scheme", "P1"]
        text_reliability.main()

        sys.argv = ["text_benefit", "--config", yml, "--visual-run-dir", run_dir,
                    "--text-scheme", "P1", "--folds", "3",
                    "--benefit-kappa", "0.5", "--device", "cpu"]
        text_benefit.main()
        benefit_json = os.path.join(run_dir, "text_benefit_P1.json")
        _assert_benefit_smoke(benefit_json, sizes["train"])
        _assert_sample_gate_smoke()

        logp = evaluate_fusion.prob_scores(
            torch.randn(5, C), torch.randn(5, C), torch.full((5, C), 0.3),
            Tv=1.0, Tt=1.0)
        assert torch.allclose(logp.exp().sum(dim=1), torch.ones(5), atol=1e-6)

        for mode in calibration.MODES:
            sys.argv = ["evaluate_fusion", "--config", yml, "--mode", mode,
                        "--text-scheme", "P1", "--visual-run-dir", run_dir, "--test"]
            evaluate_fusion.main()
            vt = f"{smoke_out}/VT_{mode}_P1_seed1"
            for f in ("selected_fusion.json", "val_results.json", "test_results.json"):
                assert os.path.exists(os.path.join(vt, f)), f"{mode}: missing {f}"
            if mode == "adaptive":
                sys.argv = ["evaluate_fusion", "--config", yml,
                            "--load-best-config",
                            os.path.join(vt, "selected_fusion.json"),
                            "--eval-split", "test"]
                evaluate_fusion.main()

        prob_runs = [
            ("prob_fixed", ["--alpha-maxes", "0.2,0.4"]),
            ("prob_G", ["--alpha-maxes", "0.2", "--etas", "0.5"]),
            ("prob_B", ["--alpha-maxes", "0.2", "--tau-Bs", "0.5"]),
            ("prob_GB", ["--alpha-maxes", "0.2", "--etas", "0.5",
                         "--tau-Bs", "0.5"]),
            ("prob_B_sample", ["--alpha-maxes", "0.0,0.2", "--tau-Bs", "0.5"]),
            ("prob_GB_sample", ["--alpha-maxes", "0.0,0.2", "--etas", "0.5",
                                "--tau-Bs", "0.5"]),
            ("prob_GQR", ["--alpha-maxes", "0.2", "--gammas", "0.5",
                          "--etas", "0.5"]),
        ]
        for mode, extra in prob_runs:
            sys.argv = ["evaluate_fusion", "--config", yml, "--mode", mode,
                        "--text-scheme", "P1", "--visual-run-dir", run_dir,
                        "--test"] + extra
            evaluate_fusion.main()
            vt = f"{smoke_out}/VT_{mode}_P1_seed1"
            for f in ("selected_fusion.json", "val_results.json", "test_results.json"):
                assert os.path.exists(os.path.join(vt, f)), f"{mode}: missing {f}"
            if mode in ("prob_B", "prob_GB", "prob_B_sample", "prob_GB_sample"):
                assert os.path.exists(os.path.join(vt, "val_class_diagnostics.csv"))
            if mode in ("prob_B_sample", "prob_GB_sample"):
                selected = utils.load_json(os.path.join(vt, "selected_fusion.json"))
                assert selected["selected"]["benefit_application"] == "sample_expectation"
                with open(os.path.join(vt, "val_diagnostics.csv"), newline="") as f:
                    header = next(csv.reader(f))
                for col in ("sample_text_confidence", "sample_alpha",
                            "text_top_class", "text_top_probability", "B_true"):
                    assert col in header, f"{mode}: missing diagnostic column {col}"

        vt = f"{smoke_out}/VT_prob_GB_sample_P1_seed1"
        sys.argv = ["evaluate_fusion", "--config", yml,
                    "--load-best-config", os.path.join(vt, "selected_fusion.json"),
                    "--eval-split", "test"]
        evaluate_fusion.main()

        print(f"\nSMOKE TEST PASSED - visual test groupAvgAcc={avg:.2f} "
              f"(chance={100.0/C:.1f}); text benefit + logit/prob fusion ran end-to-end.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(smoke_out, ignore_errors=True)


if __name__ == "__main__":
    main()
