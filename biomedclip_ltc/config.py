"""Config loading for the BiomedCLIP add-on.

Reuses MONICA's YAML Config (utils/setup_configs.Config) so head/medium/tail
cutoffs, numpy paths, img_path, num_classes, dataset_name, etc. come from the
SAME source of truth as the baselines. Adds a `biomedclip` section.

Run as a module from the MONICA repo root so `utils` is importable:
    python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/isic_100.yml
"""
from utils.setup_configs import Config

# Defaults filled in if absent from the YAML `biomedclip:` section.
_DEFAULTS = {
    "hub_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "feature_dim": 512,
    "feature_root": "./biomedclip_ltc/features",
    "proto_root": "./biomedclip_ltc/text_prototypes",
    "isic_label_map": "./biomedclip_ltc/isic_label_map.json",
    # metadata CSV is used ONLY for data verification + the Stage-0 lesion_id
    # leakage check. It NEVER enters a prompt or the model.
    "isic_metadata_csv": "./ISIC_2019_Training_Metadata.csv",
    "text_scheme": "P1",             # P0 name-only | P1 modality template | P2 fine-grained clinical
    "variant": "V",
    # fusion / calibration
    "reliability_temperature": 0.1,
    "visual_temperature": 1.0,
    # optimization (Stage-2 visual head defaults, updated from experiments)
    "lr": 0.01,
    "weight_decay": 0.0,
    "epochs": 50,
    "batch_size": 256,
    "cos_lr": False,
    "extract_batch_size": 128,
    "seeds": [1, 2, 3],
    "lt_loss": "CE",
}


def load_config(yml_path):
    """Load a MONICA-style YAML and ensure cfg.biomedclip has all defaults."""
    cfg = Config(yml_path)
    if cfg.biomedclip is None:
        cfg.biomedclip = Config()
    for k, v in _DEFAULTS.items():
        if k not in cfg.biomedclip:
            cfg.biomedclip[k] = v
    return cfg


def bm(cfg, key):
    """Read a biomedclip field with default fallback."""
    val = cfg.biomedclip[key] if key in cfg.biomedclip else None
    return _DEFAULTS[key] if val is None else val
