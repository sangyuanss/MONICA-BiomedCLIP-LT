"""Calibration signals for adaptive visual+text fusion (Stage 5).

- visual_uncertainty: normalized entropy of the visual prediction (0..1).
- sample_text_reliability: how trustworthy the text branch is for THIS sample,
  = sum_c p_text(c) * class_text_reliability(c).
- fusion_alpha: per-sample text weight under each ablation mode.

These are pure functions on tensors; no training, no test leakage.
"""
import math

import torch
import torch.nn.functional as F

MODES = ("fixed", "uncertainty_only", "reliability_only", "adaptive")


def visual_uncertainty(visual_logits, eps=1e-8):
    """Normalized entropy of softmax(visual_logits) -> [N] in [0,1].
    High = visual head unsure (let text help more)."""
    p = F.softmax(visual_logits, dim=1)
    ent = -(p * (p + eps).log()).sum(dim=1)
    return ent / math.log(visual_logits.shape[1])


def sample_text_reliability(text_logits, class_reliability):
    """[N]: expected class text-reliability under the text branch's own
    distribution. class_reliability: [C] in [0,1]."""
    p_text = F.softmax(text_logits, dim=1)
    return (p_text * class_reliability.to(text_logits.device)).sum(dim=1)


def fusion_alpha(mode, lam, uncertainty=None, sample_reliability=None, n=None,
                 alpha_fixed=None, device=None):
    """Per-sample text weight alpha_i (shape [N]).

      fixed             -> alpha_fixed (constant, broadcast)
      uncertainty_only  -> lam * u_i
      reliability_only  -> lam * s_i
      adaptive          -> lam * u_i * s_i
    """
    if mode == "fixed":
        assert alpha_fixed is not None and n is not None
        return torch.full((n,), float(alpha_fixed), device=device)
    if mode == "uncertainty_only":
        return lam * uncertainty
    if mode == "reliability_only":
        return lam * sample_reliability
    if mode == "adaptive":
        return lam * uncertainty * sample_reliability
    raise ValueError(f"unknown fusion mode '{mode}' (use {MODES})")
