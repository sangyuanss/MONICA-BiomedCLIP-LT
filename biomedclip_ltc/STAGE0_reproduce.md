# Stage 0 — MONICA baseline reproduction & add-on bootstrap

Everything here runs from the **MONICA repo root** on the GPU server (AutoDL 4090
24G). The add-on never modifies MONICA; it reuses `numpy/` splits, `dic.npy`
labels, and `utils/log_accuracy.py` metrics.

## 0.1 Environment

```bash
# MONICA's own env (PyTorch + timm + sklearn ...)
conda env create -f MONICA.yml
conda activate MONICA

# extra dependency for the BiomedCLIP add-on
pip install open_clip_torch
```

The add-on also uses `randaugment` indirectly only through MONICA's data pipeline
(not needed for feature extraction). Stages 1–6 of the add-on do **not** import
MONICA's training loop, so missing `randaugment` will not affect them.

## 0.2 Data layout to set on the server

Edit `img_path` in the configs to point at the real image roots (these match the
MONICA defaults):

| dataset | config `img_path` | split files |
|---|---|---|
| ISIC-2019-LT | `/path/to/isic2019/train/` (files like `ISIC_0000000.jpg`) | `numpy/isic/{train,val,test}_100.npy`, `numpy/isic/dic.npy` |
| KVASIR-LT | `/path/to/gastrointestinal/labeled-images/` (paths like `lower-gi-tract/.../x.jpg`) | `numpy/kvasir/{train,val,test}_20.npy`, `numpy/kvasir/dic.npy` |

Also place `ISIC_2019_Training_GroundTruth.csv` and `ISIC_2019_Training_Metadata.csv`
(from https://challenge.isic-archive.com/data/#2019) on the server for verification
and the leakage check.

## 0.3 Reproduce MONICA baselines (sanity, optional but recommended)

```bash
# ISIC IR=100 ERM baseline (ResNet-50, timm)
python main.py --config ./configs/isic/100/isic_ERM.yml
# KVASIR IR=20 ERM baseline
python main.py --config ./configs/kvasir/20/kvasir_ERM.yml
```
Outputs land in `outputs/<dataset>/<save_name>/logs.txt` with per-epoch
`group acc` = `[head, medium, tail, avg]` and the same for AUROC/AUPRC/F1.
`best.pt` is selected on **val** group-avg acc (`accuracy[3]`); test is logged
every epoch but never drives selection. The add-on copies this exact protocol.

Group cutoffs (from the configs, indices into the frequency-sorted classes):
- ISIC: `head=2` → {0,1}; `medium=5` → {2,3,4}; tail → {5,6,7}.
- KVASIR: `head=4` → {0-3}; `medium=8` → {4-7}; tail → {8-13}.

## 0.4 Add-on Stage 0 — verify ISIC mapping + leakage report

```bash
# 1) verify ISIC label -> diagnosis (writes biomedclip_ltc/isic_label_map.json)
python -m biomedclip_ltc.verify_isic_mapping \
    --gt /path/to/ISIC_2019_Training_GroundTruth.csv

# 2) cross-split leakage reports (writes biomedclip_ltc/stage0_leakage_report_*.json)
python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/isic_100.yml \
    --metadata-csv /path/to/ISIC_2019_Training_Metadata.csv
python -m biomedclip_ltc.leakage_check --config biomedclip_ltc/configs/kvasir_20.yml
```

### Verified ISIC label map (100% purity, from the official GroundTruth)
| label | abbrev | full name | dic # | train_LT (IR=100) |
|---|---|---|---|---|
| 0 | NV | melanocytic nevus | 12875 | 5000 |
| 1 | MEL | melanoma | 4522 | 2590 |
| 2 | BCC | basal cell carcinoma | 3323 | 1342 |
| 3 | BKL | benign keratosis | 2624 | 695 |
| 4 | AK | actinic keratosis | 867 | 360 |
| 5 | SCC | squamous cell carcinoma | 628 | 187 |
| 6 | VASC | vascular lesion | 253 | 97 |
| 7 | DF | dermatofibroma | 239 | 51 |

### Leakage findings (already run locally on the split files)
- **Exact image-name overlap** train/val/test: **CLEAN** for both ISIC and KVASIR.
- **ISIC lesion_id cross-split: ⚠ LEAKAGE** — 492 lesions span >1 split, 1,637
  images affected (923 images have no lesion_id). This is inherent to MONICA's
  published IR=100 split and affects all MONICA ISIC baselines.
  **Decision pending** (see DESIGN.md §13): keep official split + disclose, or
  build a lesion-disjoint re-split.

## 0.5 Add-on Stage 1 — feature & prototype extraction (next step)

```bash
python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/isic_100.yml --split all
python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/kvasir_20.yml --split all
```
Produces (see DESIGN.md §12): `features/<ds>/<split>_<IR>_{feats,labels}.npy`,
`text_prototypes/<ds>_{P0,P1,P2}.npy`, and `features/<ds>/extract_meta.json`
(checkpoint, transform, tokenizer, prompt version, splits, label map,
normalization). ISIC extraction **aborts** unless `isic_label_map.json` exists.

Stages 2–6 (visual head, text-only, fusion, gate, ablations + final test) are
implemented in subsequent submissions.
