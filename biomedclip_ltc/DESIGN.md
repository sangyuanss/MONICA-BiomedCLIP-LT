# Reliability-Aware Text-Calibrated Long-Tailed Medical Classification

**A frozen-BiomedCLIP add-on for MONICA.** Candidate contribution — *not yet a confirmed
innovation*. This document is the design contract; nothing here claims novelty until the
ablation ladder (Section 7) demonstrates that the gains come specifically from the **joint**
modeling of visual uncertainty and text reliability.

---

## 0. One-paragraph summary

We freeze the BiomedCLIP image and text encoders and operate entirely on **offline-cached
features**. A linear visual head produces visual logits `z_v`. Class text prototypes produce
zero-shot text logits `z_t`. For each sample we measure **visual prediction entropy** `u` (how
unsure the visual head is) and for each class we measure a **train-set image↔text similarity
margin** `r` (how reliable that class's text prototype is). A small **reliability-aware
calibration gate** turns `(u, r)` into a per-sample-per-class weight `α` that controls how much
`z_t` compensates `z_v`. Final logits `z_f = z_v + α ⊙ z_t`. The encoders are never trained; the
test split never participates in reliability estimation, gate/head training, or hyper-parameter
/ checkpoint selection.

---

## 1. Why a separate offline pipeline (and not a new MONICA `method`)

- The brief mandates a **frozen BiomedCLIP offline-feature mode**. With frozen encoders the
  expensive part (image encoding) is computed **once** and cached; training the head + gate is
  then a fast operation on `D`-dim vectors.
- MONICA's `main.py` train loop is built around timm backbones, online augmentation, and a
  per-method dispatch. Wedging BiomedCLIP into it would re-encode every image every epoch and
  risk touching baseline code paths. The brief explicitly requires **not breaking the original
  MONICA baselines**.
- Therefore this lives in a self-contained sub-package `MONICA/biomedclip_ltc/` that **reuses**
  MONICA's data splits (`numpy/`), label dict (`dic.npy`), and Head/Medium/Tail metric code
  (`utils/log_accuracy.py`) but does **not** modify `main.py`, `models/`, `losses/`, or any
  existing config.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `C` | number of classes (ISIC=8, KVASIR=14) |
| `D` | BiomedCLIP embedding dim (512 for ViT-B/16 BiomedCLIP) |
| `x ∈ R^D` | L2-normalized frozen image embedding of a sample |
| `t_c ∈ R^D` | L2-normalized text prototype of class `c` |
| `z_v ∈ R^C` | visual logits, `z_v = W_v x + b_v` |
| `z_t ∈ R^C` | text logits, `z_{t,c} = s · (x · t_c)` (cosine × logit-scale) |
| `u ∈ [0,1]` | per-sample visual prediction entropy (normalized) |
| `r ∈ [0,1]^C` | per-class text reliability (margin-based, train-only) |
| `α ∈ R_{≥0}^{C}` (per sample) | calibration gate output |
| `z_f` | fused logits used for prediction & loss |

BiomedCLIP source (default): `open_clip.create_model_and_transforms(
'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')`, tokenizer via
`open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')`.
Embedding dim `D=512`. All image/text embeddings are L2-normalized before use.

---

## 3. The five model components

### 3.1 Frozen BiomedCLIP feature mode (offline)
`extract_features.py` runs the **frozen** image encoder over the train/val/test names from the
MONICA numpy splits and writes one feature matrix per split (`features/<dataset>/<split>_<IR>.npy`,
shape `[N, D]`, plus an aligned label vector). It also encodes the class prompts with the
**frozen** text encoder into `text_prototypes/<dataset>.npy` (shape `[C, D]`). Encoders are put
in `.eval()` with `torch.no_grad()`; no gradients ever flow into them.

### 3.2 Visual linear classification head
`W_v: R^D → R^C` (+ bias). Trained on cached image features. Standalone, this **is** the
visual baseline (a linear probe on frozen BiomedCLIP).

### 3.3 Class text prototypes → text logits
`z_{t,c} = s · (x · t_c)`. Prototypes are an **ensemble**: each class has several prompt
templates; their text embeddings are averaged and renormalized. `s` is a logit-scale
(initialized to BiomedCLIP's, optionally learnable). Standalone, `argmax z_t` **is** the
text-only (zero-shot) baseline.

### 3.4 Visual uncertainty (entropy)
`p_v = softmax(z_v)`, `u = H(p_v) / log C ∈ [0,1]`. High `u` ⇒ the visual head is unsure ⇒ a
candidate to lean on text. `u` is **detached** before entering the gate (default) so the gate
cannot train the head toward artificial uncertainty; this choice is itself an ablation knob.

### 3.5 Reliability-aware text calibration gate
A small learnable function `gate(u, r) = sigmoid(w1·u + w2·r + w3·(u·r) + b)`, producing
`α_{i,c} = λ · gate(u_i, r_c)`. It is **sample×class-level**: it depends on the sample (via `u_i`)
and on the class (via `r_c`). `w1,w2,w3,b,λ` (and optionally `s`) are the only extra trainable
parameters. Fused logits:

```
z_{f, i, c} = z_{v, i, c} + α_{i,c} · z_{t, i, c}
```

---

## 4. Per-class text reliability `r_c` (train-only, offline)

For every **training** image `x_i` with label `y_i`:

```
margin_i = (x_i · t_{y_i})  −  max_{c ≠ y_i} (x_i · t_c)
```

Class-level raw reliability `ρ_c = mean_{i : y_i = c} margin_i`. Normalize across classes to
`r_c ∈ [0,1]`:

```
r_c = sigmoid( (ρ_c − mean_c ρ) / (std_c ρ + ε) )          # robust, default
```

(min-max normalization offered as an alternative.) `r ∈ R^C` is stored as a **fixed buffer** and
reused at val/test time **unchanged**. Intuition: a tail class with few images but a
well-separated text prototype still earns high reliability, so text is allowed to help it; a
class whose prototype is confusable earns low reliability and text is suppressed there.

> **Leakage note:** `ρ_c` uses *train* features and *labels* only. It is frozen before any val/
> test forward pass.

---

## 5. Loss, optimizer, training

- **Trainable:** `W_v, b_v`, gate `{w1,w2,w3,b}`, `λ`, optionally `s`. **Frozen:** both encoders,
  text prototypes, `r`.
- **Loss (default):** plain `CrossEntropy(z_f, y)` for **every** ladder rung, so any difference
  between rungs is attributable to the *fusion mechanism*, not to a long-tailed loss. An optional
  LT loss (Balanced-Softmax / logit-adjustment, reusing `MONICA/losses/`) is studied as a
  *separate* orthogonal add-on, never as the default that proves the contribution.
- **Optimizer:** Adam on cached features (cheap; runs on CPU or a single GPU in seconds–minutes).
- **Model selection:** checkpoint with best **validation** group-average accuracy
  (`accuracy[3]` from MONICA's `calculate_accs`) — identical criterion to MONICA's `best.pt`.
- **Test:** evaluated **once**, at the end, with the val-selected checkpoint.

---

## 6. Test-set leakage controls (hard requirements)

1. `r_c` computed from **train** features/labels only.
2. Visual head + gate trained on **train** only.
3. **All** hyper-parameters (λ range & init, gate init, lr, epochs, `s`, reliability
   normalization, prompt set) selected on **val** group-avg acc.
4. **Test** features are loaded only inside the final-evaluation function and feed nothing but
   the metric computation — never fitting, never selection, never reliability.
5. Each split's feature cache is produced independently; there is no cross-split statistic that
   touches test.

A single `assert`-guarded boundary in `train.py` (test loader constructed only after the best
checkpoint is fixed) enforces #4 in code.

---

## 7. Ablation ladder (must run in this order)

| Tag | Model | `z_f` | Trainable extras | Purpose |
|---|---|---|---|---|
| **V** | Visual baseline | `z_v` | head | frozen-BiomedCLIP linear probe |
| **T** | Text-only | `z_t` | (optional `s`) | zero-shot prototypes |
| **F-fixed** | Fixed fusion | `z_v + λ·z_t` | head, scalar `λ` | does any text help? |
| **F-class** | Class-level fusion | `z_v + λ·r_c·z_t` | head, `λ` | does *reliability* help? (no per-sample `u`) |
| **F-full** | Full adaptive fusion | `z_v + α_{i,c}·z_t` | head, gate | proposed method |

**Disentangling ablations** (to justify the "joint" claim):
- **A-unc**: `α_i = λ·sigmoid(w1·u_i + b)` — uncertainty only, no `r`.
- **A-rel**: = **F-class** — reliability only, no `u`.
- **A-full**: = **F-full** — both.

**The contribution is considered supported only if** `F-full > F-fixed > V`, `F-full ≥ T` on
tail/medium, **and** `A-full > A-unc` **and** `A-full > A-rel` — i.e. uncertainty and reliability
each contribute and the combination beats either alone. Report Head/Medium/Tail Acc + AUROC +
AUPRC + F1 (MONICA's four metrics) for every rung, mean ± std over seeds {1,2,3}.

If these inequalities do **not** hold, we report the negative result and do **not** claim
innovation.

---

## 8. Datasets & label→name maps

### ISIC-2019-LT (IR=100, 8 classes; head={0,1}, med={2,3,4}, tail={5,6,7})
Label ids are assigned by descending frequency. Canonical mapping (⚠ **confirm against the
script that built `numpy/isic/dic.npy`, or spot-check a few IDs vs the official ISIC-2019
GroundTruth before the final run**):

| id | dx | prompt class name |
|---|---|---|
| 0 | NV | melanocytic nevus |
| 1 | MEL | melanoma |
| 2 | BCC | basal cell carcinoma |
| 3 | BKL | benign keratosis |
| 4 | AK | actinic keratosis |
| 5 | SCC | squamous cell carcinoma |
| 6 | VASC | vascular lesion |
| 7 | DF | dermatofibroma |

### KVASIR-LT (HyperKvasir labeled-images, IR=20, 14 classes; head={0-3}, med={4-7}, tail={8-13})
Recovered directly from image-path folders (verified in this repo's `dic.npy`):

| id | folder | humanized prompt name |
|---|---|---|
| 0 | bbps-2-3 | colon with good bowel preparation (BBPS 2-3) |
| 1 | polyps | colorectal polyp |
| 2 | cecum | cecum |
| 3 | dyed-lifted-polyps | dyed and lifted polyp |
| 4 | pylorus | pylorus |
| 5 | dyed-resection-margins | dyed resection margin |
| 6 | z-line | esophageal z-line |
| 7 | retroflex-stomach | retroflex view of the stomach |
| 8 | bbps-0-1 | colon with poor bowel preparation (BBPS 0-1) |
| 9 | ulcerative-colitis-grade-2 | ulcerative colitis grade 2 |
| 10 | esophagitis-a | esophagitis grade A |
| 11 | retroflex-rectum | retroflex view of the rectum |
| 12 | esophagitis-b-d | esophagitis grade B-D |
| 13 | ulcerative-colitis-grade-1 | ulcerative colitis grade 1 |

Prompt templates are ensembled (e.g. dermoscopy: `"a dermoscopic image of {name}"`,
`"dermoscopy showing {name}"`, …; endoscopy: `"an endoscopic image of {name}"`,
`"gastrointestinal endoscopy showing {name}"`, …).

---

## 9. File layout (skeleton created; bodies pending approval)

```
MONICA/biomedclip_ltc/
  DESIGN.md            ← this file
  README.md            ← how to run (stage 1 extract → stage 2 train → ablations)
  __init__.py
  config.py            ← lightweight cfg loader (reuses MONICA Config) + dataclass
  configs/
    isic_100.yml
    kvasir_20.yml
  prompts/
    __init__.py
    isic.py            ← label→name + templates
    kvasir.py          ← label→name + templates
  extract_features.py  ← Stage 1: cache frozen image feats + text prototypes
  data.py              ← load cached features/labels aligned to MONICA splits
  reliability.py       ← per-class train-only margin reliability r_c
  uncertainty.py       ← visual entropy u
  model.py             ← VisualHead, TextHead, CalibrationGate, FusionModel (5 variants)
  losses.py            ← CE default (+ optional LT loss adapter to MONICA/losses)
  metrics.py           ← thin adapter to MONICA/utils/log_accuracy.calculate_metrics
  train.py             ← Stage 2: train head+gate, val-select, single test eval
  run_ablations.py     ← drives V, T, F-fixed, F-class, F-full, A-unc, A-rel over seeds
  utils.py             ← seeding, IO, logging
```

## 10. Resolved decisions (from user)
1. **ISIC label→dx mapping** — must be *verified*, not guessed. `verify_isic_mapping.py`
   cross-references `dic.npy` image names against the official
   `ISIC_2019_Training_GroundTruth.csv` and writes `isic_label_map.json`
   (`label → {abbrev, full_name, count}`). Extraction refuses to build ISIC text unless this
   verified map exists. Frequency-based guessing is **not** acceptable.
2. **BiomedCLIP** = `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, using its
   *own* `preprocess_val` image transform and `get_tokenizer`, with **L2 normalization** applied
   to both image and text embeddings. Caches record checkpoint id, feature dim, split, and the
   preprocessing string.
3. **Plain CE** for the entire ablation ladder. Balanced-Softmax / Logit-Adjustment are
   *separate orthogonal* experiments, never mixed into the core comparison.
4. **Protocol**: ISIC IR=100, KVASIR IR=20. Debug end-to-end + all ablations on **seed=1** first;
   only after the method is shown effective, run seeds {1,2,3}.
5. **val/test discipline**: val selects model + hyper-parameters during training; **test is
   evaluated exactly once** after the final model is fixed.

---

## 11. Text schemes P0 / P1 / P2 — all CLASS-LEVEL (no patient metadata)

The text channel is one **class prototype per class** (`[C, D]`); every candidate of a class
shares it, and **test time is image-only**. Three independent schemes:

- **P0 — name only**: prompt = the class name (`"melanoma"`).
- **P1 — modality template**: name in a simple template (`"a dermoscopic image of melanoma"`),
  template-ensembled.
- **P2 — fine-grained clinical (this is "Plan B")**: medically-checked descriptions of lesion
  morphology, borders, color, texture, typical structures and imaging appearance. Example:
  *"Melanoma, dermoscopic features: asymmetric lesion, irregular borders, color variegation,
  atypical pigment network, irregular dots and globules, and a blue-white veil."* ISIC uses
  **dermoscopic** feature fields; KVASIR uses **endoscopic** appearance fields (no forced shared
  phrasing). May have several prompts per class.

For every scheme the prototype is built by encoding all prompts of a class, **L2-normalizing**,
**averaging**, then **renormalizing** → `t_c`. Text logits `z_t = s·(x·t_c)`.

**Rationale for P2:** with frozen BiomedCLIP we cannot adapt image features for rare tail
classes; richer class-level clinical text widens the semantic separation between classes, which
is where the reliability margin `r_c` and the gate can help tail/medium classes.

**Defaults:** the main method and core ablations use **P0 or P1** (config `text_scheme`, default
`P1`). **P2 is an optional extension**, run explicitly. **No patient metadata** (age/sex/site/
lesion_id) ever enters a prompt. `lesion_id` is used solely for the Stage-0 leakage check
(Section 13). KVASIR has no patient metadata at all — symmetric across datasets.

Text reliability `r_c` (Section 4) and the gate (Section 3.5) sit on top of whichever scheme's
`X @ t_cᵀ` is used; all train-only.

---

## 12. Cache format (Stage 1 output)

```
biomedclip_ltc/features/<dataset>/<split>_<IR>_feats.npy    [N, D] f32, L2-normalized image emb
biomedclip_ltc/features/<dataset>/<split>_<IR>_labels.npy   [N]    i64
biomedclip_ltc/text_prototypes/<dataset>_P0.npy             [C, D] f32, L2-normalized
biomedclip_ltc/text_prototypes/<dataset>_P1.npy             [C, D] f32, L2-normalized
biomedclip_ltc/text_prototypes/<dataset>_P2.npy             [C, D] f32, L2-normalized
biomedclip_ltc/features/<dataset>/extract_meta.json         provenance
biomedclip_ltc/isic_label_map.json                          verified ISIC label→dx map
```
`extract_meta.json` records: `hub_id`, `feature_dim`, `normalization`, `logit_scale`,
`preprocess` (str of the open_clip val transform), `tokenizer` + context length,
`prompt_version`, `schemes_cached`, per-split `N`, source numpy split paths, and the verified
`isic_label_map`. Downstream stages assert these match the config before training.

---

## 13. Stage-0 data integrity (leakage report)

`leakage_check.py` writes `stage0_leakage_report_<dataset>_<IR>.json`:
1. **Exact image-name overlap** across splits — both datasets: **CLEAN** (0 overlaps).
2. **lesion_id cross-split** (ISIC, from metadata CSV; lesion_id used here ONLY): **⚠ LEAKAGE
   FOUND** in MONICA's official ISIC IR=100 split — **492 lesions span multiple splits, 1,637
   images affected** (e.g. `BCN_0001755` in train+val+test); 923 images lack a lesion_id and
   cannot be checked. This is a property of MONICA's published split and affects **all** MONICA
   ISIC baselines, not just this method.

**Open decision (does not block Stage 0/1):** either (a) keep MONICA's official split for
apples-to-apples comparison with published baselines and disclose the leakage, or (b) build a
lesion-disjoint re-split (loses direct comparability to MONICA numbers). To be decided before
the final ISIC results table.
