"""Calibration signals for adaptive visual+text fusion (Stage 5).

- visual_uncertainty: normalized entropy of the calibrated visual prediction.
- sample_text_reliability: expected train-estimated class text reliability under
  the calibrated text prediction distribution.
- fusion_alpha: per-sample text weight under each ablation mode, with validation
  gate-mean normalization and alpha clipping.

These are pure functions on tensors; no training, no test leakage.
"""
import math

import torch
import torch.nn.functional as F

MODES = ("fixed", "uncertainty_only", "reliability_only", "adaptive")


def visual_uncertainty(visual_logits, temperature=1.0, eps=1e-8):
    """Normalized entropy of softmax(visual_logits / T) -> [N] in [0,1].

    High means the visual head is unsure, so text is allowed to contribute more.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    p = F.softmax(visual_logits / float(temperature), dim=1)
    ent = -(p * p.clamp_min(eps).log()).sum(dim=1)
    return (ent / math.log(visual_logits.shape[1])).clamp(0.0, 1.0)


def sample_text_reliability(text_logits, class_reliability, text_temperature=1.0):
    """[N]: expected class text-reliability under the text branch's own
    distribution. class_reliability: [C] in [0,1]."""
    if text_temperature <= 0:
        raise ValueError(f"text_temperature must be positive, got {text_temperature}")
    p_text = F.softmax(text_logits / float(text_temperature), dim=1)
    return (p_text * class_reliability.to(text_logits.device)).sum(dim=1)


def fusion_alpha(mode, lambda_value=1.0, uncertainty=None, sample_reliability=None,
                 n=None, alpha_fixed=None, device=None, alpha_max=1.0,
                 gate_mean=None, eps=1e-8, return_gate_mean=False):
    """Per-sample text weight alpha_i (shape [N]).

      fixed             -> alpha_fixed (constant, broadcast)
      uncertainty_only  -> clip(lambda * u_i / mean_val(u), 0, alpha_max)
      reliability_only  -> clip(lambda * r_i / mean_val(r), 0, alpha_max)
      adaptive          -> clip(lambda * u_i*r_i / mean_val(u*r), 0, alpha_max)

    During validation, call with gate_mean=None and save the returned gate mean.
    During test, pass the saved validation gate_mean unchanged.
    """
    if alpha_max <= 0:
        raise ValueError(f"alpha_max must be positive, got {alpha_max}")
    if mode == "fixed":
        if alpha_fixed is None:
            alpha_fixed = lambda_value
        assert n is not None
        alpha = torch.full((n,), float(alpha_fixed), device=device)
        alpha = alpha.clamp(0.0, float(alpha_max))
        return (alpha, 1.0) if return_gate_mean else alpha
    if mode == "uncertainty_only":
        if uncertainty is None:
            raise ValueError("uncertainty is required for uncertainty_only")
        gate = uncertainty
    elif mode == "reliability_only":
        if sample_reliability is None:
            raise ValueError("sample_reliability is required for reliability_only")
        gate = sample_reliability
    elif mode == "adaptive":
        if uncertainty is None or sample_reliability is None:
            raise ValueError("uncertainty and sample_reliability are required for adaptive")
        gate = uncertainty * sample_reliability
    else:
        raise ValueError(f"unknown fusion mode '{mode}' (use {MODES})")

    used_gate_mean = float(gate.mean().clamp_min(eps).item()) if gate_mean is None else float(gate_mean)
    alpha = float(lambda_value) * gate / (used_gate_mean + eps)
    alpha = alpha.clamp(0.0, float(alpha_max))
    return (alpha, used_gate_mean) if return_gate_mean else alpha
