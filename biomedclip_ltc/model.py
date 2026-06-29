"""Models for the BiomedCLIP add-on.

Stage 2 ships the visual linear classification head (the 'V' visual baseline):
a single linear layer on top of FROZEN, cached BiomedCLIP image features. The
BiomedCLIP encoder is never part of this module (features are precomputed in
Stage 1), so 'freezing the encoder' is structural — only this head has params.

Text head, calibration gate, and the fusion variants arrive in Stage 5.
"""
import torch.nn as nn


class VisualHead(nn.Module):
    """Linear visual classifier on frozen features: z_v = W_v x (+ b_v)."""

    def __init__(self, feature_dim, num_classes, bias=True):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)
