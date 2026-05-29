"""
train.py
========

Training script for HWS-Mamba on EchoNet-Pediatric / EchoNet-Dynamic.

Expected dataset layout (one CSV manifest per dataset):

    data/echonet_peds_a4c/
        clips/              # *.npy or *.avi/*.mp4 clips
        masks/              # *.npy ED/ES masks
        manifest.csv        # columns: clip_id, split, ed_idx, es_idx, ef

Run:
    python train.py --data-root data/echonet_peds_a4c --epochs 300
                    --batch-size 16 --resolution 256 --output runs/peds_a4c
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from hws_mamba import build_hws_mamba
from loss import HWSLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    clip_id: str
    split: str
    ed_idx: int
    es_idx: int
    ef: float
    has_ef: bool   # False for PSAX (no EF supervision)


class EchoClipDataset(Dataset):
    """Loads (clip, mask, ed_idx, es_idx, ef, has_ef) tuples.

    Clips are expected as (T_full, H, W) grayscale numpy arrays in [0, 1].
    Masks are expected as (T_full, H, W) binary arrays (annotated only on
    ED and ES frames; intermediate frames are zero, see HSS-Net protocol).

    The loader uniformly samples T frames so that the ED frame is the
    first and the ES frame is the last, and resizes to `resolution`.
    """

    def __init__(
        self,
        root: str,
        split: str,
        resolution: int = 256,
        clip_len: int = 10,
        augment: bool = False,
        has_ef_view: bool = True,
    ):
        self.root = Path(root)
        self.resolution = resolution
        self.clip_len = clip_len
        self.augment = augment
        self.has_ef_view = has_ef_view

        manifest = self.root / "manifest.csv"
        self.records = []
        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                self.records.append(ClipRecord(
                    clip_id=row["clip_id"],
                    split=row["split"],
                    ed_idx=int(row["ed_idx"]),
                    es_idx=int(row["es_idx"]),
                    ef=float(row.get("ef", 0.0)) / 100.0,
                    has_ef=has_ef_view,
                ))

    def __len__(self):
        return len(self.records)

    def _load_clip(self, clip_id: str) -> np.ndarray:
        return np.load(self.root / "clips" / f"{clip_id}.npy")

    def _load_mask(self, clip_id: str) -> np.ndarray:
        return np.load(self.root / "masks" / f"{clip_id}.npy")

    def _sample_indices(self, T_full: int, ed: int, es: int) -> np.ndarray:
        ed, es = min(ed, es), max(ed, es)
        ed, es = max(0, ed), min(T_full - 1, es)
        if es - ed + 1 < self.clip_len:
            es = min(T_full - 1, ed + self.clip_len - 1)
        return np.linspace(ed, es, self.clip_len).round().astype(int)

    @staticmethod
    def _resize(x: np.ndarray, size: int) -> np.ndarray:
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(t.shape[-3], size, size),
                          mode="trilinear" if t.ndim == 5 else "bilinear",
                          align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    def _augment(self, clip, mask):
        # Mild augmentations (each ~0.5 prob)
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            clip = np.clip(clip ** gamma, 0, 1)
        if np.random.rand() < 0.5:
            clip = np.clip(clip + np.random.uniform(-0.05, 0.05), 0, 1)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            clip = np.clip(clip * scale, 0, 1)
        return clip, mask

    def __getitem__(self, i: int):
        rec = self.records[i]
        clip = self._load_clip(rec.clip_id).astype(np.float32)
        mask = self._load_mask(rec.clip_id).astype(np.float32)

        idx = self._sample_indices(clip.shape[0], rec.ed_idx, rec.es_idx)
        clip = clip[idx]   # (T, H, W)
        mask = mask[idx]   # (T, H, W)

        clip = self._resize(clip, self.resolution)
        mask = (self._resize(mask, self.resolution) > 0.5).astype(np.float32)

        if self.augment:
            clip, mask = self._augment(clip, mask)

        clip = torch.from_numpy(clip).unsqueeze(0)   # (1, T, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)   # (1, T, H, W)

        return {
            "clip":   clip,
            "mask":   mask,
            "ed_idx": torch.tensor(0,                  dtype=torch.long),
            "es_idx": torch.tensor(self.clip_len - 1,  dtype=torch.long),
            "ef":     torch.tensor(rec.ef,             dtype=torch.float32),
            "has_ef": torch.tensor(float(rec.has_ef),  dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for batch in loader:
        clip   = batch["clip"].to(device)
        mask   = batch["mask"].to(device)
        ed_idx = batch["ed_idx"].to(device)
        es_idx = batch["es_idx"].to(device)
        ef_gt  = batch["ef"].to(device)
        has_ef = batch["has_ef"].to(device)

        logits = model(clip)
        loss = criterion(logits, mask, ed_idx, es_idx, ef_gt, use_ef=has_ef)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * clip.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    for batch in loader:
        clip   = batch["clip"].to(device)
        mask   = batch["mask"].to(device)
        ed_idx = batch["ed_idx"].to(device)
        es_idx = batch["es_idx"].to(device)
        ef_gt  = batch["ef"].to(device)
        has_ef = batch["has_ef"].to(device)
        logits = model(clip)
        loss = criterion(logits, mask, ed_idx, es_idx, ef_gt, use_ef=has_ef)
        running += loss.item() * clip.size(0)
    return running / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",  required=True, type=str)
    p.add_argument("--output",     required=True, type=str)
    p.add_argument("--epochs",     type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--clip-len",   type=int, default=10)
    p.add_argument("--lr-max",     type=float, default=1e-4)
    p.add_argument("--lr-min",     type=float, default=1e-5)
    p.add_argument("--alpha",      type=float, default=0.8)
    p.add_argument("--lambda-ef",  type=float, default=0.5)
    p.add_argument("--has-ef-view", type=int, default=1,
                   help="0 for PSAX (no EF), 1 for A4C-based datasets.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume",     type=str, default="")
    args = p.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = EchoClipDataset(args.data_root, "train",
                                resolution=args.resolution,
                                clip_len=args.clip_len, augment=True,
                                has_ef_view=bool(args.has_ef_view))
    val_set   = EchoClipDataset(args.data_root, "val",
                                resolution=args.resolution,
                                clip_len=args.clip_len, augment=False,
                                has_ef_view=bool(args.has_ef_view))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    model = build_hws_mamba(num_classes=1).to(device)
    criterion = HWSLoss(alpha=args.alpha, lambda_ef=args.lambda_ef).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr_max)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr_min)

    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr\n")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"epoch {epoch+1:03d}/{args.epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.2e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{lr_now:.2e}\n")

        ckpt = dict(model=model.state_dict(), optim=optimizer.state_dict(),
                    sched=scheduler.state_dict(), epoch=epoch,
                    best_val=best_val, args=vars(args))
        torch.save(ckpt, out_dir / "last.pth")
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, out_dir / "best.pth")


if __name__ == "__main__":
    main()
