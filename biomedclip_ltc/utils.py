"""Seeding, IO, and image-loading helpers.

set_seed mirrors MONICA/main.py for comparability. Image loaders mirror
MONICA/dataset/dataloader.py so the same files resolve identically.
"""
import json
import os
import random

import numpy as np
from PIL import Image


def set_seed(seed=1, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def pil_loader(img_path, name):
    """Open img_path + name as RGB (matches MONICA default_loader)."""
    return Image.open(img_path + name).convert("RGB")


def load_split_names_labels(np_path, dict_path):
    """Return (names list, labels np.int64 array) for a MONICA numpy split."""
    dic = np.load(dict_path, allow_pickle=True).item()
    names = list(np.load(np_path, allow_pickle=True))
    labels = np.array([dic[n] for n in names], dtype=np.int64)
    return names, labels


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def method_dir(dataset, name):
    """Standard output directory for a method run: outputs/biomedclip_ltc/<ds>/<name>."""
    return os.path.join("outputs", "biomedclip_ltc", dataset, name)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
