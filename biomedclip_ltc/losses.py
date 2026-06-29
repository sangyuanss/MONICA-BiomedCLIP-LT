"""Loss factory. Default = plain CrossEntropy (so ladder gains attribute to the
fusion mechanism, not to a long-tailed loss).

Balanced-Softmax and Logit-Adjustment are implemented as REAL callables for the
*separate, orthogonal* experiments only — they are never the default and are not
mixed into the core method comparison.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmax(nn.Module):
    """Ren et al., 2020: logits + log(class_prior) before CE."""

    def __init__(self, cls_num_list):
        super().__init__()
        freq = torch.tensor(cls_num_list, dtype=torch.float)
        self.register_buffer("log_prior", torch.log(freq / freq.sum() + 1e-12))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_prior.to(logits.device), target)


class LogitAdjust(nn.Module):
    """Menon et al., 2021: logits + tau * log(class_prior) before CE."""

    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        freq = torch.tensor(cls_num_list, dtype=torch.float)
        self.register_buffer("log_prior", torch.log(freq / freq.sum() + 1e-12))
        self.tau = tau

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.tau * self.log_prior.to(logits.device), target)


def get_loss(cfg, cls_num_list=None):
    """Return a loss callable(logits, targets). Default 'CE'."""
    from biomedclip_ltc import config as cfgmod
    name = cfgmod.bm(cfg, "lt_loss")
    if name == "CE":
        return nn.CrossEntropyLoss()
    if name == "BalancedSoftmax":
        assert cls_num_list is not None, "BalancedSoftmax needs cls_num_list"
        return BalancedSoftmax(cls_num_list)
    if name == "LogitAdjust":
        assert cls_num_list is not None, "LogitAdjust needs cls_num_list"
        return LogitAdjust(cls_num_list)
    raise ValueError(f"unknown lt_loss '{name}' (use CE | BalancedSoftmax | LogitAdjust)")
