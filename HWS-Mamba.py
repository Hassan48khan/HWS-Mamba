"""
HWS-Mamba: Hierarchical Windowed Spatio-temporal Mamba
for Pediatric Left Ventricle Segmentation and EF Estimation.

Architecture origin (strictly from two papers, nothing else):

    [HSS-Net]   - Hierarchical design: convolutional blocks at low-level
                  stages (single-frame detail) + Mamba blocks at
                  high-level stages (multi-frame spatio-temporal).
                - Differentiable EF training signal alongside Dice/BCE.

    [WAS-Mamba] - Cross-channel Window Scan with half-window shifting,
                  preserving local neighbourhoods when a feature map is
                  flattened into a 1D sequence.
                - Weighted State-Space Module (WSSM) that fuses spatial
                  and frequency-domain embeddings to re-weight the four
                  scan branches before fusion.

HWS-Mamba simply combines and adapts these to the 2D+t pediatric LV
segmentation problem (both A4C and PSAX, same weights).  No new module
families beyond what already appears in HSS-Net or WAS-Mamba.

Concrete adaptations
--------------------
1. WSTCS  : WAS-Mamba's CCWScan operates on 3D volumes by walking
            (h, w, c) inside a 2D spatial window across channels.  For
            2D+t echo video we walk (t, h, w) inside a space-time window.
            We keep the four-direction structure (forward/backward x
            original/shifted) and the half-window torch.roll shifting.
2. WSSM2D+t : WAS-Mamba's WSSM with the same spatial + frequency dual
              embedding, applied to 2D+t feature maps instead of 3D
              volumes.  No additional embedding branch is introduced.
3. EF loss : HSS-Net is trained with Dice + BCE.  Their results show
             good Dice but unsatisfactory EF.  We add a differentiable
             Simpson EF loss term alongside Dice + BCE.  No new module.

Notes
-----
- The CPU fallback path (no `mamba_ssm` available) is a pass-through so
  that the file remains importable for shape and gradient sanity tests.
  On GPU with the standard Mamba kernel the SSM behaves normally.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_MAMBA = True
except Exception:  # pragma: no cover
    selective_scan_fn = None
    _HAS_MAMBA = False


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _to_3tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)


class DropPath(nn.Module):
    """Stochastic depth per sample, as in timm."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


# ---------------------------------------------------------------------------
# Patch embedding / merging / expanding for 2D+t video
# ---------------------------------------------------------------------------

class VideoPatchEmbed(nn.Module):
    """Split a (B, C, T, H, W) clip into non-overlapping patches.
    Returns (B, T', H', W', C_embed)."""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 64,
        norm_layer: Optional[Callable] = nn.LayerNorm,
    ):
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class SpatialPatchMerging(nn.Module):
    """Halve H and W, double C.  T preserved."""

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class SpatialPatchExpand(nn.Module):
    """Inverse: double H and W, halve C."""

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = rearrange(x, 'b t h w (p1 p2 c) -> b t (h p1) (w p2) c',
                      p1=2, p2=2)
        return self.norm(x)


class FinalPatchExpand(nn.Module):
    """Restore (H, W) back to input resolution at the end of the decoder."""

    def __init__(self, dim: int, scale: int = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = scale
        self.expand = nn.Linear(dim, scale * scale * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = rearrange(x, 'b t h w (p1 p2 c) -> b t (h p1) (w p2) c',
                      p1=self.scale, p2=self.scale)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Low-level block: separable convolution (HSS-Net low-level stage)
# ---------------------------------------------------------------------------

class SepConvBlock(nn.Module):
    """Inverted residual separable conv block applied per-frame, then a
    feed-forward layer.  This is HSS-Net's low-level Stage 1/Stage 2
    block, with the same Layer-Norm -> SeparableConv -> FFN structure."""

    def __init__(
        self,
        dim: int,
        expand: int = 4,
        drop_path: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        hidden = dim * expand
        self.norm1 = norm_layer(dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.norm2 = norm_layer(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop_path = DropPath(drop_path)

    def _per_frame_conv(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        y = self.dw(y); y = self.pw1(y); y = self.act(y); y = self.pw2(y)
        return y.permute(0, 2, 3, 1).contiguous().reshape(B, T, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self._per_frame_conv(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Windowed Spatio-Temporal Cross Scan (WSTCS)
# Adaptation of WAS-Mamba's CCWScan from 3D volumes to 2D+t video.
# ---------------------------------------------------------------------------

def _make_window_index(T, H, W, Tw, Hw, Ww, device) -> torch.Tensor:
    """Permutation of length T*H*W that re-orders a (t, h, w)-flattened
    tensor so all tokens of one window are contiguous, windows are
    visited in (Tw, Hw, Ww) raster order, and inside each window tokens
    are in (t, h, w) order.  Direct analogue of WAS-Mamba's window
    indexing, lifted to a space-time window."""
    assert T % Tw == 0 and H % Hw == 0 and W % Ww == 0
    nh, nw = H // Hw, W // Ww
    t = torch.arange(T, device=device)
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')
    orig_pos = (tt * H + hh) * W + ww
    wt, it = tt // Tw, tt % Tw
    wh, ih = hh // Hw, hh % Hw
    ww_, iw = ww // Ww, ww % Ww
    window_id = (wt * nh + wh) * nw + ww_
    inside_id = (it * Hw + ih) * Ww + iw
    new_pos = window_id * (Tw * Hw * Ww) + inside_id
    order = torch.argsort(new_pos.reshape(-1))
    return orig_pos.reshape(-1)[order].long()


class WSTCS(nn.Module):
    """Windowed Spatio-Temporal Cross Scan.

    Same idea as WAS-Mamba's CCWScan: split the feature map into windows,
    scan inside each window, scan the windows in raster order, and run a
    second scan on a half-window-shifted copy to recover features split
    by the first windowing.  Difference: the window is 3D in space-time
    (Tw, Hw, Ww), not (Hw, Ww, channels), because here we are processing
    2D+t video.

    Produces four scan paths (stacked as (B, 4, C, L)):
        P1: forward windowed scan on the original clip
        P2: backward windowed scan on the original clip
        P3: forward windowed scan on the half-window shifted clip
        P4: backward windowed scan on the shifted clip
    """

    def __init__(self, window: Tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.window = window
        self._cache: dict = {}

    def _get_index(self, T, H, W, device):
        Tw, Hw, Ww = self.window
        Tw = min(Tw, T); Hw = min(Hw, H); Ww = min(Ww, W)
        Tw = max(1, T // max(1, T // Tw))
        Hw = max(1, H // max(1, H // Hw))
        Ww = max(1, W // max(1, W // Ww))
        key = (T, H, W, Tw, Hw, Ww, str(device))
        if key not in self._cache:
            self._cache[key] = _make_window_index(T, H, W, Tw, Hw, Ww, device)
        return self._cache[key], (Tw, Hw, Ww)

    @staticmethod
    def _shift(x: torch.Tensor, win) -> torch.Tensor:
        Tw, Hw, Ww = win
        return torch.roll(x, shifts=(Tw // 2, Hw // 2, Ww // 2),
                          dims=(2, 3, 4))

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        idx, win = self._get_index(T, H, W, x.device)
        L = T * H * W
        flat = x.reshape(B, C, L)
        p1 = flat.index_select(dim=-1, index=idx)
        p2 = torch.flip(p1, dims=(-1,))
        shifted = self._shift(x, win).reshape(B, C, L)
        p3 = shifted.index_select(dim=-1, index=idx)
        p4 = torch.flip(p3, dims=(-1,))
        return torch.stack([p1, p2, p3, p4], dim=1), idx, win

    def unscan(self, seqs, idx, win, T, H, W) -> torch.Tensor:
        B, K, C, L = seqs.shape
        assert K == 4 and L == T * H * W
        s = [seqs[:, 0], torch.flip(seqs[:, 1], dims=(-1,)),
             seqs[:, 2], torch.flip(seqs[:, 3], dims=(-1,))]
        out = seqs.new_empty(B, K, C, L)
        for j in range(4):
            out[:, j].index_copy_(dim=-1, index=idx, source=s[j])
        out = out.reshape(B, K, C, T, H, W)
        Tw, Hw, Ww = win
        out_s = torch.roll(out[:, 2:],
                           shifts=(-(Tw // 2), -(Hw // 2), -(Ww // 2)),
                           dims=(3, 4, 5))
        return torch.cat([out[:, :2], out_s], dim=1)


# ---------------------------------------------------------------------------
# Weighted State-Space Module (WSSM) - WAS-Mamba's WSSM ported to 2D+t.
# ---------------------------------------------------------------------------

class WSSM2Dt(nn.Module):
    """WAS-Mamba's Weighted State-Space Module, ported from 3D volumes to
    2D+t echo video.

    Faithful port: still uses the spatial-domain depthwise prior, the
    frequency-domain (FFT) embedding, fuses them into a single hybrid
    embedding, runs the four scan branches through a packed selective
    scan, and re-weights each branch with the hybrid embedding before
    summing them.  Only the tensor rank changes (T, H, W instead of
    D, H, W) and the scan module is the new WSTCS.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        window: Tuple[int, int, int] = (2, 4, 4),
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 32) if dt_rank == "auto" else dt_rank

        # Input projection (x and gate z), as in WAS-Mamba.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True)

        # Spatial branch: depthwise 3D conv image prior.
        self.dw_conv = nn.Conv3d(self.d_inner, self.d_inner,
                                 kernel_size=d_conv, padding=d_conv // 2,
                                 groups=self.d_inner, bias=True)

        # Frequency branch: FFT -> 1x1 conv -> iFFT, as in WAS-Mamba.
        self.fft_proj = nn.Conv3d(self.d_inner, self.d_inner,
                                  kernel_size=1, bias=True)

        # Fuse spatial + frequency into the hybrid embedding used to
        # re-weight the four scan branches (exactly the WSSM design).
        self.hybrid_fuse = nn.Sequential(
            nn.Conv3d(2 * self.d_inner, self.d_inner,
                      kernel_size=d_conv, padding=d_conv // 2,
                      groups=self.d_inner, bias=True),
            nn.BatchNorm3d(self.d_inner),
            nn.SiLU(inplace=True),
        )

        self.scan = WSTCS(window=window)

        # Selective-scan parameters, packed for 4 branches.
        K = 4
        self.x_proj_weight = nn.Parameter(torch.empty(
            K, self.dt_rank + 2 * d_state, self.d_inner))
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        dt_projs = [self._init_dt_proj(dt_init, dt_scale,
                                       dt_min, dt_max, dt_init_floor)
                    for _ in range(K)]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([p.weight for p in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(
            torch.stack([p.bias for p in dt_projs], dim=0))

        self.A_logs = self._init_A_log(d_state, self.d_inner, copies=K)
        self.Ds = self._init_D(self.d_inner, copies=K)

        # Per-branch reweighting using the hybrid embedding (WSSM design).
        self.branch_weight = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(2 * self.d_inner, self.d_inner,
                          kernel_size=d_conv, padding=d_conv // 2,
                          groups=self.d_inner, bias=True),
                nn.BatchNorm3d(self.d_inner),
                nn.SiLU(inplace=True),
            ) for _ in range(K)
        ])

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ---- parameter init (kept identical to WAS-Mamba's WSSM) -------------

    def _init_dt_proj(self, dt_init, dt_scale, dt_min, dt_max, floor):
        dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -std, std)
        else:
            nn.init.constant_(dt_proj.weight, std)
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _init_A_log(d_state, d_inner, copies):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> r d n", r=copies).flatten(0, 1)
        p = nn.Parameter(A_log); p._no_weight_decay = True
        return p

    @staticmethod
    def _init_D(d_inner, copies):
        D = torch.ones(d_inner)
        D = repeat(D, "n -> r n", r=copies).flatten(0, 1)
        p = nn.Parameter(D); p._no_weight_decay = True
        return p

    # ---- the two image-prior branches and the selective scan -------------

    def _hybrid_embedding(self, x_main: torch.Tensor) -> torch.Tensor:
        """Spatial + frequency embedding, as in WAS-Mamba's WSSM."""
        x_sp = self.dw_conv(x_main)

        freq = torch.fft.fftn(x_main, dim=(2, 3, 4))
        freq = self.fft_proj(freq.real) + 1j * self.fft_proj(freq.imag)
        x_fr = torch.fft.ifftn(freq, dim=(2, 3, 4)).real

        return self.hybrid_fuse(torch.cat([x_sp, x_fr], dim=1))

    def _selective_scan_branches(self, xs: torch.Tensor) -> torch.Tensor:
        B, K, C, L = xs.shape
        x_dbl = torch.einsum("bkcl,krc->bkrl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        xs_f = xs.reshape(B, K * C, L).float()
        dts_f = dts.reshape(B, K * C, L).float()
        Bs_f = Bs.reshape(B, K, self.d_state, L).float()
        Cs_f = Cs.reshape(B, K, self.d_state, L).float()
        Ds = self.Ds.float()
        As = -torch.exp(self.A_logs.float())
        dt_bias = self.dt_projs_bias.reshape(-1).float()

        if _HAS_MAMBA:
            out = selective_scan_fn(
                xs_f, dts_f, As, Bs_f, Cs_f, Ds, z=None,
                delta_bias=dt_bias, delta_softplus=True,
                return_last_state=False,
            )
        else:
            out = xs_f  # CPU pass-through for sanity testing
        return out.view(B, K, C, L)

    # ---- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C)."""
        B, T, H, W, C = x.shape

        xz = self.in_proj(x)
        x_in, z_in = xz.chunk(2, dim=-1)
        x_main = x_in.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,T,H,W)

        hybrid = self._hybrid_embedding(x_main)

        x_conv = F.silu(self.dw_conv(x_main))
        seqs, idx, win = self.scan(x_conv)
        y_seqs = self._selective_scan_branches(seqs)
        y_imgs = self.scan.unscan(y_seqs, idx, win, T, H, W)

        outs = [self.branch_weight[k](
                    torch.cat([hybrid, y_imgs[:, k]], dim=1))
                for k in range(4)]
        y = sum(outs)

        z = z_in.permute(0, 4, 1, 2, 3).contiguous()
        y = y * F.silu(z)
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = self.out_norm(y)
        return self.dropout(self.out_proj(y))


# ---------------------------------------------------------------------------
# High-level block (HSS-Net's Stage 3/4 + WAS-Mamba's WSSM)
# ---------------------------------------------------------------------------

class STMambaBlock(nn.Module):
    """HSS-Net's high-level spatio-temporal block, with WSSM2D+t as the
    sequence mixer.  Pre-norm Mamba + pre-norm FFN."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path: float = 0.0,
        window: Tuple[int, int, int] = (2, 4, 4),
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ssm = WSSM2Dt(d_model=dim, d_state=d_state, expand=expand,
                           window=window)
        self.norm2 = norm_layer(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ssm(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------

class EncoderStage(nn.Module):
    def __init__(self, dim, depth, block_type, drop_path,
                 d_state=16, window=(2, 4, 4), downsample=None):
        super().__init__()
        self.block_type = block_type
        if block_type == 'sepconv':
            self.blocks = nn.ModuleList([
                SepConvBlock(dim=dim, drop_path=drop_path[i])
                for i in range(depth)])
        elif block_type == 'mamba':
            self.blocks = nn.ModuleList([
                STMambaBlock(dim=dim, d_state=d_state,
                             drop_path=drop_path[i], window=window)
                for i in range(depth)])
        else:
            raise ValueError(block_type)
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class DecoderStage(nn.Module):
    def __init__(self, dim, depth, block_type, drop_path,
                 d_state=16, window=(2, 4, 4), upsample=None):
        super().__init__()
        self.block_type = block_type
        self.upsample = upsample(dim=dim) if upsample is not None else None
        out_dim = dim // 2 if upsample is not None else dim
        if block_type == 'sepconv':
            self.blocks = nn.ModuleList([
                SepConvBlock(dim=out_dim, drop_path=drop_path[i])
                for i in range(depth)])
        elif block_type == 'mamba':
            self.blocks = nn.ModuleList([
                STMambaBlock(dim=out_dim, d_state=d_state,
                             drop_path=drop_path[i], window=window)
                for i in range(depth)])
        else:
            raise ValueError(block_type)

    def forward(self, x, skip):
        if skip is not None:
            x = x + skip
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Full HWS-Mamba network
# ---------------------------------------------------------------------------

class HWSMamba(nn.Module):
    """HWS-Mamba: Hierarchical Windowed Spatio-temporal Mamba for
    pediatric LV segmentation and EF estimation.

    Encoder (after HSS-Net):
        Stage 1, 2  - SepConvBlock  (per-frame, local detail)
        Stage 3, 4  - STMambaBlock  (multi-frame, WSTCS + WSSM2D+t)
    Decoder symmetric, with skip connections.
    Output: per-frame LV segmentation logits.

    The same weights handle both A4C and PSAX clips - there is no
    view-specific code path.
    """

    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 1,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        depths_decoder: Sequence[int] = (2, 2, 2, 2),
        d_state: int = 16,
        drop_path_rate: float = 0.1,
        patch_size: Tuple[int, int, int] = (1, 4, 4),
        windows: Sequence[Tuple[int, int, int]] = ((2, 4, 4),) * 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.dims = dims

        self.patch_embed = VideoPatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        dpr = list(torch.linspace(0, drop_path_rate, sum(depths)).tolist())
        cur = 0
        self.enc1 = EncoderStage(
            dim=dims[0], depth=depths[0], block_type='sepconv',
            drop_path=dpr[cur:cur + depths[0]],
            downsample=lambda dim: SpatialPatchMerging(dim)
        ); cur += depths[0]
        self.enc2 = EncoderStage(
            dim=dims[1], depth=depths[1], block_type='sepconv',
            drop_path=dpr[cur:cur + depths[1]],
            downsample=lambda dim: SpatialPatchMerging(dim)
        ); cur += depths[1]
        self.enc3 = EncoderStage(
            dim=dims[2], depth=depths[2], block_type='mamba',
            drop_path=dpr[cur:cur + depths[2]],
            d_state=d_state, window=windows[0],
            downsample=lambda dim: SpatialPatchMerging(dim)
        ); cur += depths[2]
        self.enc4 = EncoderStage(
            dim=dims[3], depth=depths[3], block_type='mamba',
            drop_path=dpr[cur:cur + depths[3]],
            d_state=d_state, window=windows[1],
            downsample=None,
        )

        dpr_dec = list(reversed(dpr))
        cur = 0
        self.dec4 = DecoderStage(
            dim=dims[3], depth=depths_decoder[0], block_type='mamba',
            drop_path=dpr_dec[cur:cur + depths_decoder[0]],
            d_state=d_state, window=windows[2],
            upsample=lambda dim: SpatialPatchExpand(dim)
        ); cur += depths_decoder[0]
        self.dec3 = DecoderStage(
            dim=dims[2], depth=depths_decoder[1], block_type='mamba',
            drop_path=dpr_dec[cur:cur + depths_decoder[1]],
            d_state=d_state, window=windows[3],
            upsample=lambda dim: SpatialPatchExpand(dim)
        ); cur += depths_decoder[1]
        self.dec2 = DecoderStage(
            dim=dims[1], depth=depths_decoder[2], block_type='sepconv',
            drop_path=dpr_dec[cur:cur + depths_decoder[2]],
            upsample=lambda dim: SpatialPatchExpand(dim)
        ); cur += depths_decoder[2]
        self.dec1 = DecoderStage(
            dim=dims[0], depth=depths_decoder[3], block_type='sepconv',
            drop_path=dpr_dec[cur:cur + depths_decoder[3]],
            upsample=None,
        )

        self.final_up = FinalPatchExpand(dim=dims[0], scale=patch_size[1])
        self.seg_head = nn.Conv3d(dims[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, num_classes, T, H, W) logits."""
        f = self.patch_embed(x)
        f, s1 = self.enc1(f)
        f, s2 = self.enc2(f)
        f, s3 = self.enc3(f)
        f, _ = self.enc4(f)
        f = self.dec4(f, None)
        f = self.dec3(f, s3)
        f = self.dec2(f, s2)
        f = self.dec1(f, s1)
        f = self.final_up(f).permute(0, 4, 1, 2, 3).contiguous()
        return self.seg_head(f)


# ---------------------------------------------------------------------------
# Losses: HSS-Net's Dice + BCE, augmented with a differentiable Simpson
# EF loss.  No new module - just an additional scalar in the loss.
# ---------------------------------------------------------------------------

def soft_dice_loss(pred, target, eps=1e-6):
    dims = (2, 3, 4)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def simpson_single_plane_ef(mask: torch.Tensor,
                            ed_idx: torch.Tensor,
                            es_idx: torch.Tensor) -> torch.Tensor:
    """Differentiable Simpson single-plane EF surrogate.

    Per-frame "volume" is approximated by sum of cubed row sums of the
    mask - a smooth analogue of the disk-summation rule.  Monotone in
    the true EF, fully differentiable, suitable as a training signal.
    """
    B, _, T, H, W = mask.shape
    row_sums = mask.sum(dim=-1)
    volume = (row_sums ** 3).sum(dim=-1).squeeze(1)         # (B, T)
    b = torch.arange(B, device=mask.device)
    v_ed = volume[b, ed_idx]
    v_es = volume[b, es_idx]
    ef = (v_ed - v_es) / (v_ed + 1e-6)
    return ef.clamp(0.0, 1.0)


class HWSLoss(nn.Module):
    """L_total = a * L_dice + (1 - a) * L_bce + lambda_ef * L_ef.

    HSS-Net uses a * Dice + (1-a) * BCE with a = 0.8.  We retain that
    and add a small Simpson-EF term.
    """

    def __init__(self, alpha: float = 0.8, lambda_ef: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_ef = lambda_ef
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, ed_idx, es_idx, ef_gt):
        prob = torch.sigmoid(logits)
        l_dice = soft_dice_loss(prob, target)
        l_bce = self.bce(logits, target)
        ef_pred = simpson_single_plane_ef(prob, ed_idx, es_idx)
        l_ef = (ef_pred - ef_gt).abs().mean()
        return self.alpha * l_dice + (1 - self.alpha) * l_bce + \
            self.lambda_ef * l_ef


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Small config for CPU sanity test.
    B, C, T, H, W = 1, 1, 8, 64, 64
    net = HWSMamba(
        in_chans=C, num_classes=1, embed_dim=32,
        depths=(2, 2, 2, 2), depths_decoder=(2, 2, 2, 2),
        d_state=8, drop_path_rate=0.0,
        patch_size=(1, 4, 4),
        windows=((2, 4, 4),) * 4,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"HWS-Mamba (small) parameters: {n_params/1e6:.2f} M")

    x = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        y = net(x)
    print("input ", tuple(x.shape))
    print("output", tuple(y.shape))
    assert y.shape == (B, 1, T, H, W)

    # Realistic config parameter count.
    big = HWSMamba(
        in_chans=1, num_classes=1, embed_dim=64,
        depths=(2, 2, 2, 2), depths_decoder=(2, 2, 2, 2),
        d_state=16, drop_path_rate=0.1,
    )
    print(f"HWS-Mamba (realistic) parameters: "
          f"{sum(p.numel() for p in big.parameters())/1e6:.2f} M")

    # Loss sanity check.
    target = (torch.rand_like(y) > 0.5).float()
    ed_idx = torch.tensor([0])
    es_idx = torch.tensor([T - 1])
    ef_gt = torch.tensor([0.55])
    loss = HWSLoss(alpha=0.8, lambda_ef=0.5)(y, target, ed_idx, es_idx, ef_gt)
    print(f"loss = {loss.item():.4f}")
