"""ISIC-2019-LT prompt registry. Three independent, CLASS-LEVEL text schemes.

  P0 : class NAME only.
  P1 : class name + a simple modality template (dermoscopy).
  P2 : fine-grained, medically-checked CLASS-LEVEL clinical descriptions
       (lesion morphology, borders, color, dermoscopic structures). May have
       several prompts per class; their L2-normalized text features are averaged
       into one class prototype. This is "Plan B".

All schemes are class-level: every candidate of a class shares the same text
prototype, and at test time ONLY the image is the input. No patient metadata
(age/sex/site/lesion_id) ever enters a prompt — the metadata CSV is used only for
data verification and the Stage-0 leakage check.

Class names come from the VERIFIED map (isic_label_map.json). Use
load_label_map(..., strict=True) for real extraction.
"""
import json
import os
import warnings

PROMPT_VERSION = "isic-v2"  # P0/P1/P2 class-level (Plan B = fine-grained clinical)

_FALLBACK = [
    ("NV", "melanocytic nevus"), ("MEL", "melanoma"),
    ("BCC", "basal cell carcinoma"), ("BKL", "benign keratosis"),
    ("AK", "actinic keratosis"), ("SCC", "squamous cell carcinoma"),
    ("VASC", "vascular lesion"), ("DF", "dermatofibroma"),
]

# P1 modality templates (name only).
TEMPLATES = [
    "a dermoscopic image of {}",
    "dermoscopy showing {}",
    "a skin lesion image of {}",
    "this is a dermoscopic image of {}, a type of skin lesion",
]

# P2 fine-grained dermoscopic descriptions per official dx abbreviation.
# One or more prompts per class (averaged after L2-norm). PLEASE REVIEW clinically.
FINE_GRAINED = {
    "NV": [
        "Melanocytic nevus, dermoscopic features: symmetric lesion, regular "
        "well-defined borders, uniform coloration, a typical regular pigment "
        "network, and evenly distributed dots and globules.",
    ],
    "MEL": [
        "Melanoma, dermoscopic features: asymmetric lesion, irregular borders, "
        "color variegation, atypical pigment network, irregular dots and "
        "globules, and a blue-white veil.",
    ],
    "BCC": [
        "Basal cell carcinoma, dermoscopic features: absent pigment network, "
        "arborizing branching vessels, blue-gray ovoid nests, leaf-like areas, "
        "and ulceration.",
    ],
    "BKL": [
        "Benign keratosis, dermoscopic features: milia-like cysts, comedo-like "
        "openings, a cerebriform brain-like surface, and sharply demarcated "
        "borders.",
    ],
    "AK": [
        "Actinic keratosis, dermoscopic features: an erythematous background, a "
        "strawberry pattern, white-to-yellow keratotic scale, and follicular "
        "openings surrounded by white halos.",
    ],
    "SCC": [
        "Squamous cell carcinoma, dermoscopic features: keratin masses, white "
        "circles around hair follicles, looped and glomerular vessels, and "
        "central ulceration or scale.",
    ],
    "VASC": [
        "Vascular lesion, dermoscopic features: well-defined red, purple or "
        "blue lacunae separated by pale fibrous septa.",
    ],
    "DF": [
        "Dermatofibroma, dermoscopic features: a central white scar-like patch, "
        "a delicate peripheral pigment network, and a firm fibrous nodule.",
    ],
}


def load_label_map(path="./biomedclip_ltc/isic_label_map.json", strict=False):
    """Return list[(abbrev, full_name)] by label id. strict=True forbids fallback."""
    if os.path.exists(path):
        data = json.load(open(path))
        labels = data["labels"]
        out = [None] * len(labels)
        for k, v in labels.items():
            out[int(k)] = (v["abbrev"], v["full_name"])
        if any(o is None for o in out):
            raise ValueError(f"{path} has gaps in label ids: {list(labels)}")
        return out
    if strict:
        raise FileNotFoundError(
            f"Verified ISIC label map {path} not found. Run "
            "`python -m biomedclip_ltc.verify_isic_mapping --gt <GroundTruth.csv>` "
            "before caching ISIC text features.")
    warnings.warn(f"[ISIC] {path} missing — using FREQUENCY-GUESS fallback (dry test only).")
    return list(_FALLBACK)


def class_names(label_map=None):
    label_map = label_map or load_label_map()
    return [full for _, full in label_map]


def build(scheme, label_map=None):
    """Return list[list[str]] of prompts per class for scheme in {P0, P1, P2}."""
    label_map = label_map or load_label_map()
    out = []
    for ab, full in label_map:
        if scheme == "P0":
            out.append([full])
        elif scheme == "P1":
            out.append([t.format(full) for t in TEMPLATES])
        elif scheme == "P2":
            fg = FINE_GRAINED.get(ab)
            if not fg:
                raise KeyError(f"No P2 fine-grained prompt for ISIC class {ab}")
            out.append(list(fg))
        else:
            raise ValueError(f"unknown scheme {scheme}")
    return out
