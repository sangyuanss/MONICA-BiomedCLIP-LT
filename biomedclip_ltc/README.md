# biomedclip_ltc

Frozen-BiomedCLIP offline-feature add-on for MONICA: reliability-aware text
calibration for long-tailed medical image classification. **Candidate method —
see `DESIGN.md`; no innovation is claimed until the ablation ladder passes.**

This sub-package does not modify MONICA. It reuses `numpy/` splits, `dic.npy`
labels, and `utils/log_accuracy.py` metrics. **Run everything from the MONICA repo
root.**

## Install

```bash
conda activate MONICA              # MONICA's env
pip install -r biomedclip_ltc/requirements.txt   # adds open_clip_torch, etc.
```

### Cache paths via env vars (recommended on servers; avoids editing tracked YAML)
```bash
export BIOMEDCLIP_FEATURE_ROOT=/root/autodl-tmp/feature_cache/isic_ir100/features
export BIOMEDCLIP_PROTO_ROOT=/root/autodl-tmp/feature_cache/isic_ir100/text_prototypes
# optional: export BIOMEDCLIP_ISIC_METADATA_CSV=/path/ISIC_2019_Training_Metadata.csv
```
These override `feature_root`/`proto_root` in the YAML, so you never have to modify
config files on the server (no `git pull` conflicts).

## Smoke test (CPU, no GPU / images / BiomedCLIP needed)

Validates the whole train -> evaluate_test path on a synthetic cache:

```bash
python -m biomedclip_ltc.smoke_test
```

## Stage 0 — verify ISIC mapping + leakage report

```bash
python -m biomedclip_ltc.verify_isic_mapping --gt /path/ISIC_2019_Training_GroundTruth.csv
python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/isic_100.yml \
    --metadata-csv /path/ISIC_2019_Training_Metadata.csv
python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/kvasir_20.yml
```
ISIC map verified at 100% purity (see `STAGE0_reproduce.md`). ⚠ The official ISIC
split has lesion-level leakage; recorded as an internal note, not blocking.

## Stage 1 — frozen feature + P0/P1/P2 prototype caching (GPU)

Set `img_path` in the configs to your image roots first.

```bash
python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/isic_100.yml --split all
python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/kvasir_20.yml --split all
```
Caches `features/<ds>/<split>_<IR>_{feats,labels}.npy`,
`text_prototypes/<ds>_{P0,P1,P2}.npy`, `features/<ds>/extract_meta.json`.

## Stage 2 — visual linear-head baseline ('V')

Train (val-selected; never touches test):
```bash
python -m biomedclip_ltc.train --config biomedclip_ltc/configs/isic_100.yml \
    --seed 1 --lr 1e-3 --batch-size 512 --epochs 100 --weight-decay 0.0
python -m biomedclip_ltc.train --config biomedclip_ltc/configs/kvasir_20.yml \
    --seed 1 --lr 1e-3 --batch-size 512 --epochs 100 --weight-decay 0.0
```
Final test (single evaluation, only after training):
```bash
python -m biomedclip_ltc.evaluate_test --run-dir outputs/biomedclip_ltc/isic/<run_name>
```
The trainer prints the exact `--run-dir` to use. Outputs (under `outputs/`, gitignored):
`best.pt`, `logs.txt`, `config_snapshot.json`, `per_class_val_best.json`,
`test_results.json`.

## Stage 3 — text-only branch (no training)

```bash
for s in P0 P1 P2; do
  python -m biomedclip_ltc.evaluate_text --config biomedclip_ltc/configs/isic_100.yml --text-scheme $s
done
# final test (after the val temperature is fixed):
python -m biomedclip_ltc.evaluate_text --config biomedclip_ltc/configs/isic_100.yml --text-scheme P1 --test
```
Outputs to `outputs/biomedclip_ltc/<ds>/T_<scheme>/`:
`selected_temperature.json`, `val_results.json`, `per_class_val.json`.

## Stage 5a — class text reliability (train-only, needed by reliability/adaptive)

```bash
python -m biomedclip_ltc.text_reliability --config biomedclip_ltc/configs/isic_100.yml --all
```
Uses the Stage-3 selected text temperature by default. Saves
`<proto_root>/<ds>_text_reliability_<scheme>.json`,
`<proto_root>/<ds>_class_reliability_<scheme>.pt`, and
`<proto_root>/<ds>_reliability_metadata_<scheme>.json`.

## Stage 4 & 5 — visual+text fusion (four ablation modes)

```bash
VRUN=outputs/biomedclip_ltc/isic/V_CE_seed1_lr0.01_bs256_ep50   # Stage-2 run dir
mkdir -p logs/fusion_grid
for s in P0 P1 P2; do
  for m in fixed uncertainty_only reliability_only adaptive; do
    python -u -m biomedclip_ltc.evaluate_fusion \
      --config biomedclip_ltc/configs/isic_100.yml \
      --mode $m --text-scheme $s --visual-run-dir $VRUN \
      2>&1 | tee logs/fusion_grid/${s}_${m}.log
  done
done
```
`fixed` searches alpha in `[0, 1]`; gated modes search lambda in `[0, 1]`.
Outputs to `VT_<mode>_<scheme>_seed<seed>/`: `selected_fusion.json`,
`val_results.json`, `per_class_val.json`, and `val_diagnostics.csv`.

Updated protocol: gated modes now search lambda in `[0, 1]`, normalize the raw
gate by its validation mean, clip with `alpha_max=1.0`, and write
`val_diagnostics.csv`. For the strict final test, load the saved validation
configuration instead of searching again:

```bash
python -u -m biomedclip_ltc.evaluate_fusion \
  --config biomedclip_ltc/configs/isic_100.yml \
  --load-best-config outputs/biomedclip_ltc/isic/VT_adaptive_P1_seed1/selected_fusion.json \
  --eval-split test
```

Probability-domain low-cost ablations (no full retraining): the current paper
experiment focuses on `prob_fixed`, `prob_G`, `prob_B`, and `prob_GB`. These mix
`softmax(z_v/Tv)` and `softmax(z_t/Tt)` with a bounded sample-class gate. `G` is
validation-quantile visual uncertainty; `B` is a train-only out-of-fold estimate
of the actual class-level text benefit. Selection ranks candidates that keep
AUROC/AUPRC within 1 point of the visual validation baseline ahead of candidates
that do not.

```bash
mkdir -p logs/fusion_prob
for s in P0 P1; do
  python -u -m biomedclip_ltc.text_benefit \
    --config biomedclip_ltc/configs/isic_100.yml \
    --visual-run-dir $VRUN \
    --text-scheme $s \
    --folds 5 \
    --benefit-kappa 0.5

  for m in prob_fixed prob_G prob_B prob_GB; do
    python -u -m biomedclip_ltc.evaluate_fusion \
      --config biomedclip_ltc/configs/isic_100.yml \
      --mode $m --text-scheme $s --visual-run-dir $VRUN \
      2>&1 | tee logs/fusion_prob/${s}_${m}.log
  done
done
```

Older Q/R probability modes (`prob_Q`, `prob_R`, `prob_GQ`, `prob_GR`,
`prob_QR`, `prob_GQR`) are still importable and runnable for reproducing the
previous long-tail/reliability ablations.

## Full method list (debug on seed=1; final report seeds 1,2,3 mean±std)

| method | command |
|---|---|
| V-CE | `train ... --lt-loss CE` |
| V-BalancedSoftmax | `train ... --lt-loss BalancedSoftmax` |
| V-LogitAdjust | `train ... --lt-loss LogitAdjust` |
| T-P0/P1/P2 | `evaluate_text --text-scheme P{0,1,2}` |
| V+T-Fixed | `evaluate_fusion --mode fixed` |
| V+T-Uncertainty | `evaluate_fusion --mode uncertainty_only` |
| V+T-Reliability | `evaluate_fusion --mode reliability_only` |
| V+T-Adaptive | `evaluate_fusion --mode adaptive` |

Visual long-tail baselines (same params as the tuned CE run):
```bash
for L in CE BalancedSoftmax LogitAdjust; do
  python -m biomedclip_ltc.train --config biomedclip_ltc/configs/isic_100.yml --lt-loss $L
done
```

## Text schemes (`text_scheme`; all class-level, image-only at test)
`P0` class name only · `P1` name + modality template (default) ·
`P2` fine-grained clinical descriptions. No patient metadata in prompts.

## Git (push to YOUR fork, not upstream PyJulie/MONICA)

```bash
# one-time: point origin at your own repo
git remote set-url origin https://github.com/<you>/MONICA.git   # or `git remote add myfork ...`
git checkout -b biomedclip-ltc

# stop tracking committed bytecode (optional cleanup)
git rm -r --cached --quiet **/__pycache__ 2>/dev/null || true

git add .gitignore biomedclip_ltc
git commit -m "Add frozen-BiomedCLIP LT add-on: Stage 0 (verify+leakage) & Stage 1-2"
git push -u origin biomedclip-ltc
```
Feature caches (`features/`, `text_prototypes/`) and `outputs/` are gitignored;
`isic_label_map.json` and the leakage report are committed as records.

## Status
- Stage 0 (verify + leakage), Stage 1 (extract), Stage 2 (visual head + test),
  Stage 3 (text-only), Stage 4 (fixed fusion), Stage 5 (reliability + uncertainty
  + adaptive fusion, 4 ablation modes): implemented and CPU-smoke-tested.
- Stage 6 (multi-seed aggregation + final results table): forthcoming.
- Visual-head defaults updated from experiments: lr=0.01, bs=256, ep=50, cos_lr=false.
