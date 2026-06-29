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
Saves `<proto_root>/<ds>_text_reliability_<scheme>.json` (reliability vector, raw
margins, and the full class×class similarity matrix for heatmaps).

## Stage 4 & 5 — visual+text fusion (four ablation modes)

```bash
VRUN=outputs/biomedclip_ltc/isic/V_CE_seed1_lr0.01_bs256_ep50   # Stage-2 run dir
for m in fixed uncertainty_only reliability_only adaptive; do
  python -m biomedclip_ltc.evaluate_fusion --config biomedclip_ltc/configs/isic_100.yml \
      --mode $m --text-scheme P1 --visual-run-dir $VRUN --test
done
```
`fixed` searches alpha∈[0..1]; the others search lambda∈[0.1,0.25,0.5,1,2,5].
Outputs to `VT_<mode>_<scheme>_seed<seed>/`: `selected_fusion.json`,
`val_results.json`, `per_class_val.json`, `test_results.json`.

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
