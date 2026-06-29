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

## Text schemes (`text_scheme`; all class-level, image-only at test)
`P0` class name only · `P1` name + modality template (default) ·
`P2` fine-grained clinical descriptions (optional). No patient metadata in prompts.

## Fusion variants (`variant`)
`V` visual baseline (Stage 2). `T / F-fixed / F-class / F-full / A-unc / A-rel`
arrive in Stages 3-6.

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
- Stage 0 (verify + leakage), Stage 1 (extract), Stage 2 (visual head + test): done.
- Stages 3-6: forthcoming.
