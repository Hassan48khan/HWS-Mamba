"""
test.py
=======

Evaluation script for HWS-Mamba.

Computes per-clip metrics on the test split and writes:
    - <output>/per_case_results.csv   (one row per clip)
    - <output>/summary.txt            (aggregate Dice, IoU, HD95, EF metrics)

Run:
    python test.py --data-root data/echonet_peds_a4c
                   --checkpoint runs/peds_a4c/best.pth
                   --output     runs/peds_a4c/eval
                   --has-ef-view 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from hws_mamba import build_hws_mamba
from loss import simpson_single_plane_ef
from train import EchoClipDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixel coordinates (N, 2) from a 2D binary mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int64)
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1]
    # boundary = mask AND NOT eroded(mask)
    eroded = (
        np.roll(mask,  1, 0) & np.roll(mask, -1, 0) &
        np.roll(mask,  1, 1) & np.roll(mask, -1, 1) & mask
    )
    boundary = mask & ~eroded
    return np.argwhere(boundary)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """HD95 (in pixels) on a 2D mask pair.  Returns NaN if either is empty."""
    p, g = _boundary_points(pred), _boundary_points(gt)
    if p.size == 0 or g.size == 0:
        return float("nan")
    d_pg = np.linalg.norm(p[:, None] - g[None, :], axis=-1).min(axis=1)
    d_gp = np.linalg.norm(g[:, None] - p[None, :], axis=-1).min(axis=1)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    """Run inference and collect per-clip metrics."""
    model.eval()
    rows = []
    for batch in loader:
        clip   = batch["clip"].to(device)
        mask   = batch["mask"].cpu().numpy()
        ed_idx = batch["ed_idx"].to(device)
        es_idx = batch["es_idx"].to(device)
        ef_gt  = batch["ef"].cpu().numpy()
        has_ef = batch["has_ef"].cpu().numpy()

        logits = model(clip)
        prob = torch.sigmoid(logits)
        ef_pred = simpson_single_plane_ef(prob, ed_idx, es_idx).cpu().numpy()
        pred = (prob.cpu().numpy() > threshold).astype(np.uint8)
        # pred / mask: (B, 1, T, H, W)

        B, _, T, H, W = pred.shape
        for b in range(B):
            ed = int(ed_idx[b].item()); es = int(es_idx[b].item())
            # Evaluate Dice/IoU/HD95 on the ED and ES frames (the annotated
            # ones); aggregate by mean.
            dices, ious, hds = [], [], []
            for fidx in (ed, es):
                p2 = pred[b, 0, fidx]
                g2 = (mask[b, 0, fidx] > 0.5).astype(np.uint8)
                dices.append(dice_score(p2, g2))
                ious.append(iou_score(p2, g2))
                hds.append(hd95(p2, g2))
            rows.append(dict(
                dice = float(np.mean(dices)),
                iou  = float(np.mean(ious)),
                hd95 = float(np.nanmean(hds)),
                ef_pred = float(ef_pred[b]),
                ef_gt   = float(ef_gt[b]),
                has_ef  = float(has_ef[b]),
            ))
    return rows


def summarise(rows, has_ef_view: bool):
    dice = np.array([r["dice"] for r in rows])
    iou  = np.array([r["iou"]  for r in rows])
    hd   = np.array([r["hd95"] for r in rows])
    out = {
        "n":         len(rows),
        "dice_mean": float(np.nanmean(dice)),
        "iou_mean":  float(np.nanmean(iou)),
        "hd95_mean": float(np.nanmean(hd)),
    }
    if has_ef_view:
        ef_pred = np.array([r["ef_pred"] for r in rows]) * 100.0  # to %
        ef_gt   = np.array([r["ef_gt"]   for r in rows]) * 100.0
        diff = ef_pred - ef_gt
        out["ef_bias"] = float(diff.mean())
        out["ef_std"]  = float(diff.std(ddof=1))
        if _HAS_SCIPY:
            r, _ = pearsonr(ef_pred, ef_gt)
            out["ef_corr"] = float(r) * 100.0
        else:
            out["ef_corr"] = float(np.corrcoef(ef_pred, ef_gt)[0, 1]) * 100.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   required=True, type=str)
    p.add_argument("--checkpoint",  required=True, type=str)
    p.add_argument("--output",      required=True, type=str)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--resolution",  type=int, default=256)
    p.add_argument("--clip-len",    type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--has-ef-view", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_set = EchoClipDataset(args.data_root, "test",
                               resolution=args.resolution,
                               clip_len=args.clip_len, augment=False,
                               has_ef_view=bool(args.has_ef_view))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    model = build_hws_mamba(num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')}).")

    rows = evaluate(model, test_loader, device)

    # Per-case CSV
    csv_path = out_dir / "per_case_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Aggregate summary
    summary = summarise(rows, has_ef_view=bool(args.has_ef_view))
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
