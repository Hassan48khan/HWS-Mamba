"""
loss.py
=======

Loss functions for HWS-Mamba: soft Dice + BCE + differentiable Simpson EF.

    L_total = alpha * L_dice + (1 - alpha) * L_bce + lambda_ef * L_ef

HSS-Net uses alpha * Dice + (1 - alpha) * BCE with alpha = 0.8; we keep
this and add a small Simpson-EF term aligned with the clinical metric.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


def soft_dice_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss over a 2D+t mask of shape (B, C, T, H, W).

    Args:
        pred:   predicted probabilities in [0, 1].
        target: binary ground-truth mask, same shape as pred.
        eps:    smoothing for numerical stability.
    """
    dims = (2, 3, 4)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def simpson_single_plane_ef(mask: torch.Tensor,
                            ed_idx: torch.Tensor,
                            es_idx: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Simpson single-plane EF surrogate.

    Each frame's "volume" is approximated by the sum of cubed row-sums of
    the mask, a smooth analogue of the disk-summation rule.  Monotone in
    the true EF and fully differentiable, suitable as a training signal.

    Args:
        mask:   (B, 1, T, H, W) predicted probabilities in [0, 1].
        ed_idx: (B,) long indices of end-diastolic frames.
        es_idx: (B,) long indices of end-systolic frames.

    Returns:
        (B,) predicted EF values in [0, 1].
    """
    B = mask.shape[0]
    row_sums = mask.sum(dim=-1)                       # (B, 1, T, H)
    volume = (row_sums ** 3).sum(dim=-1).squeeze(1)   # (B, T)
    b = torch.arange(B, device=mask.device)
    v_ed = volume[b, ed_idx]
    v_es = volume[b, es_idx]
    ef = (v_ed - v_es) / (v_ed + eps)
    return ef.clamp(0.0, 1.0)


class HWSLoss(nn.Module):
    """Total HWS-Mamba training loss: Dice + BCE + Simpson-EF."""

    def __init__(self, alpha: float = 0.8, lambda_ef: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_ef = lambda_ef
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        ed_idx: torch.Tensor,
        es_idx: torch.Tensor,
        ef_gt: torch.Tensor,
        use_ef: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss.

        Args:
            logits:  (B, 1, T, H, W) segmentation logits.
            target:  (B, 1, T, H, W) binary ground-truth.
            ed_idx:  (B,) end-diastolic frame indices.
            es_idx:  (B,) end-systolic frame indices.
            ef_gt:   (B,) ground-truth EF in [0, 1].
            use_ef:  (B,) optional 0/1 mask; samples with use_ef=0 skip
                     the EF term (e.g., PSAX clips, where EF is N/A).
                     If None, the EF term is applied to all samples.
        """
        prob = torch.sigmoid(logits)
        l_dice = soft_dice_loss(prob, target)
        l_bce = self.bce(logits, target)

        ef_pred = simpson_single_plane_ef(prob, ed_idx, es_idx)
        ef_diff = (ef_pred - ef_gt).abs()
        if use_ef is not None:
            denom = use_ef.sum().clamp(min=1.0)
            l_ef = (ef_diff * use_ef).sum() / denom
        else:
            l_ef = ef_diff.mean()

        return (self.alpha * l_dice
                + (1.0 - self.alpha) * l_bce
                + self.lambda_ef * l_ef)
