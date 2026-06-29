"""Stage 1 - offline frozen-BiomedCLIP feature extraction & caching.

Encodes every image named in MONICA's numpy splits with the FROZEN BiomedCLIP
image encoder (using the checkpoint's own preprocessing), L2-normalizes, and
caches features + labels. Also builds CLASS-LEVEL text prototypes for all three
schemes P0 / P1 / P2 (each [C, D], L2-normalized, multi-prompt averaged).

No patient metadata enters any prompt; the schemes are purely class-level, so
test time is image-only. Encoders are eval() + requires_grad=False; no gradients
ever flow into them. Run once per dataset; reused by every downstream rung.

Usage (from the MONICA repo root, on the GPU server):
    python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/isic_100.yml --split all
    python -m biomedclip_ltc.extract_features --config biomedclip_ltc/configs/kvasir_20.yml --split all
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from biomedclip_ltc import config as cfgmod
from biomedclip_ltc import utils
from biomedclip_ltc.prompts import get_prompt_module
from biomedclip_ltc.prompts import isic as isic_prompts

SCHEMES = ["P0", "P1", "P2"]


def load_biomedclip(hub_id, device):
    """Return (model, preprocess_val, tokenizer), frozen + eval."""
    import open_clip
    model, _pp_train, preprocess_val = open_clip.create_model_and_transforms(hub_id)
    tokenizer = open_clip.get_tokenizer(hub_id)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess_val, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, img_path, names, device, batch_size):
    """Encode names -> [N, D] L2-normalized float32 numpy."""
    feats, batch = [], []
    for i, name in enumerate(names):
        batch.append(preprocess(utils.pil_loader(img_path, name)))
        if len(batch) == batch_size or i == len(names) - 1:
            emb = model.encode_image(torch.stack(batch).to(device))
            feats.append(F.normalize(emb, dim=-1).float().cpu())
            batch = []
            if (i + 1) % (batch_size * 20) == 0 or i == len(names) - 1:
                print(f"    encoded {i + 1}/{len(names)} images", flush=True)
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=256):
    """Encode list[str] -> [M, D] L2-normalized float32 tensor (CPU)."""
    out = []
    for s in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[s:s + batch_size]).to(device)
        emb = model.encode_text(tokens)
        out.append(F.normalize(emb, dim=-1).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def build_prototypes(model, tokenizer, prompt_module, scheme, label_map, device):
    """Class-level prototypes [C, D]: per class, average L2-normed prompt
    embeddings then renormalize."""
    per_class = prompt_module.build(scheme, label_map)   # list[list[str]]
    protos = []
    for prompts in per_class:
        emb = encode_texts(model, tokenizer, prompts, device)       # [P, D]
        protos.append(F.normalize(emb.mean(dim=0, keepdim=True), dim=-1))
    return torch.cat(protos, dim=0).numpy().astype(np.float32)      # [C, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--schemes", default="P0,P1,P2",
                    help="comma list subset of P0,P1,P2 to cache")
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    dataset = cfg.general.dataset_name
    ir = cfg.datasets.imbalance_ratio
    C = cfg.general.num_classes
    hub_id = cfgmod.bm(cfg, "hub_id")
    feat_dim = cfgmod.bm(cfg, "feature_dim")
    feat_root = cfgmod.bm(cfg, "feature_root")
    proto_root = cfgmod.bm(cfg, "proto_root")
    img_path = cfg.datasets.img_path
    ebs = cfgmod.bm(cfg, "extract_batch_size")
    schemes = [s for s in args.schemes.split(",") if s in SCHEMES]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage1] dataset={dataset} IR={ir} C={C} device={device} hub={hub_id}")

    model, preprocess, tokenizer = load_biomedclip(hub_id, device)
    logit_scale = float(model.logit_scale.exp().item())
    prompt_module = get_prompt_module(dataset)

    # ISIC: REQUIRE the verified label map (no frequency fallback for real caches).
    label_map = None
    if dataset == "isic":
        label_map = isic_prompts.load_label_map(cfgmod.bm(cfg, "isic_label_map"), strict=True)

    out_dir = utils.ensure_dir(os.path.join(feat_root, dataset))
    utils.ensure_dir(proto_root)

    # --- class-level text prototypes for each requested scheme ---
    proto_files = {}
    for scheme in schemes:
        print(f"[Stage1] building {scheme} class-level prototypes...")
        protos = build_prototypes(model, tokenizer, prompt_module, scheme, label_map, device)
        assert protos.shape == (C, feat_dim), f"{scheme} proto {protos.shape}"
        fn = os.path.join(proto_root, f"{dataset}_{scheme}.npy")
        np.save(fn, protos)
        proto_files[scheme] = fn

    # --- image features per split ---
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    split_sizes = {}
    for split in splits:
        sp = cfg.datasets[split]
        names, labels = utils.load_split_names_labels(sp.np_path, sp.dict_path)
        print(f"[Stage1] encoding {split}: {len(names)} images")
        feats = encode_images(model, preprocess, img_path, names, device, ebs)
        assert feats.shape == (len(names), feat_dim), feats.shape
        np.save(os.path.join(out_dir, f"{split}_{ir}_feats.npy"), feats)
        np.save(os.path.join(out_dir, f"{split}_{ir}_labels.npy"), labels)
        split_sizes[split] = len(names)

    # --- provenance (checkpoint, transform, tokenizer, prompt version, splits,
    #     label map, normalization) ---
    utils.save_json(os.path.join(out_dir, "extract_meta.json"), {
        "hub_id": hub_id,
        "feature_dim": feat_dim,
        "normalization": "l2 (image + text)",
        "logit_scale": logit_scale,
        "preprocess": str(preprocess),
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_context_length": getattr(tokenizer, "context_length", None),
        "prompt_version": getattr(prompt_module, "PROMPT_VERSION", None),
        "schemes_cached": schemes,
        "proto_files": proto_files,
        "dataset": dataset,
        "imbalance_ratio": ir,
        "num_classes": C,
        "split_sizes": split_sizes,
        "split_sources": {s: cfg.datasets[s].np_path for s in splits},
        "isic_label_map": label_map,
    })
    print(f"[Stage1] done. caches in {out_dir}; prototypes in {proto_root}")


if __name__ == "__main__":
    main()
