"""Class-name and prompt-template registries per dataset.

Each dataset module exposes:
    class_names(label_map=None) -> list[str]          # index == MONICA label id
    build_prompts(label_map=None) -> list[list[str]]  # per-class generic prompts
ISIC additionally exposes build_metadata_prompt(...) for Plan B.
"""
from . import isic, kvasir

REGISTRY = {
    "isic": isic,
    "kvasir": kvasir,
}


def get_prompt_module(dataset_name: str):
    """Return the prompt module for a MONICA dataset_name ('isic' | 'kvasir')."""
    try:
        return REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No prompt module for dataset '{dataset_name}'. "
            f"Known: {sorted(REGISTRY)}")
