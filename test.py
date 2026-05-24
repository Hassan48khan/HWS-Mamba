"""Sanity tests for HWS-Mamba."""
import torch
from hws_mamba import WSTCS, HWSMamba, HWSLoss

print("=" * 60)
print("Test 1: WSTCS scan/unscan is invertible (all 4 branches)")
print("=" * 60)
scan = WSTCS(window=(2, 4, 4))
x = torch.randn(2, 8, 4, 16, 16)
seqs, idx, win = scan(x)
back = scan.unscan(seqs, idx, win, T=4, H=16, W=16)
for k in range(4):
    e = (back[:, k] - x).abs().max().item()
    print(f"  branch {k} round-trip max err: {e:.2e}")
    assert e < 1e-5
print("  PASS")

print()
print("=" * 60)
print("Test 2: Gradient flow")
print("=" * 60)
net = HWSMamba(
    in_chans=1, num_classes=1, embed_dim=32,
    depths=(1, 1, 1, 1), depths_decoder=(1, 1, 1, 1),
    d_state=8, drop_path_rate=0.0,
)
x = torch.randn(1, 1, 8, 64, 64, requires_grad=True)
y = net(x)
y.sum().backward()
unhit = [n for n, p in net.named_parameters()
         if p.grad is None or p.grad.abs().sum() == 0]
ssm_keywords = ('A_logs', '.Ds', 'x_proj_weight', 'dt_projs')
ssm_unhit = [n for n in unhit if any(s in n for s in ssm_keywords)]
non_ssm_unhit = [n for n in unhit if n not in ssm_unhit]
print(f"  unhit SSM params (CPU-fallback expected): {len(ssm_unhit)}")
print(f"  unhit non-SSM params: {len(non_ssm_unhit)}")
assert len(non_ssm_unhit) == 0, non_ssm_unhit
print(f"  input grad max: {x.grad.abs().max().item():.4e}")
print("  PASS")

print()
print("=" * 60)
print("Test 3: Variable input sizes")
print("=" * 60)
for shape in [(1, 1, 4, 64, 64), (2, 1, 8, 64, 64), (1, 1, 10, 128, 128)]:
    with torch.no_grad():
        y = net(torch.randn(*shape))
    assert y.shape == (shape[0], 1) + shape[2:]
    print(f"  in {shape} -> out {tuple(y.shape)}  OK")
print("  PASS")

print()
print("=" * 60)
print("Test 4: Multi-class output")
print("=" * 60)
net2 = HWSMamba(
    in_chans=1, num_classes=4, embed_dim=32,
    depths=(1, 1, 1, 1), depths_decoder=(1, 1, 1, 1),
    d_state=8,
)
with torch.no_grad():
    y = net2(torch.randn(1, 1, 8, 64, 64))
assert y.shape == (1, 4, 8, 64, 64)
print(f"  out: {tuple(y.shape)}  PASS")

print()
print("=" * 60)
print("Test 5: Loss end-to-end with EF supervision (batch=2)")
print("=" * 60)
y = net(torch.randn(2, 1, 8, 64, 64))
target = (torch.rand_like(y) > 0.5).float()
ed_idx = torch.tensor([0, 0]); es_idx = torch.tensor([7, 7])
ef_gt = torch.tensor([0.55, 0.45])
loss = HWSLoss(alpha=0.8, lambda_ef=0.5)(y, target, ed_idx, es_idx, ef_gt)
print(f"  loss: {loss.item():.4f}  PASS")

print()
print("All tests passed.")
