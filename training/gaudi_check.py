"""A Gaudi box, checked before any run: each card, HCCL between them, bf16
matmul speed, and FusedSDPA's answer (hpu_attention.py's call) against plain
causal attention's, forward and backward.

    torchrun --standalone --nproc_per_node=2 training/gaudi_check.py

The attention check is the one that matters: hpu_attention passes FusedSDPA
its arguments by position, and a wrong order trains on wrong attention
without any error.
"""

from __future__ import annotations

import time
import types

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl  # noqa: F401  (registers "hccl")
import hpu_attention
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv


def reduce_ok(device, rank: int, world: int) -> bool:
    x = torch.full((1024,), float(rank + 1), device=device)
    dist.all_reduce(x)
    htcore.mark_step()
    return bool((x.cpu() == world * (world + 1) / 2).all())


def tflops(device, n: int = 8192, reps: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    (a @ b).sum().item()                      # compile and warm up
    started = time.time()
    for _ in range(reps):
        c = a @ b
    htcore.mark_step()
    c.sum().item()
    return 2 * n**3 * reps / (time.time() - started) / 1e12


def attention_error(device, length: int = 2048) -> tuple[float, float]:
    """Max |fused - reference| of the output and of dL/dq, Qwen3-4B's head layout."""
    torch.manual_seed(0)
    heads, kv, dim = 32, 8, 128
    q = torch.randn(1, heads, length, dim)
    k = torch.randn(1, kv, length, dim)
    v = torch.randn(1, kv, length, dim)
    w = torch.randn(1, length, heads, dim)
    scale = dim ** -0.5

    qr = q.clone().requires_grad_()
    ref = F.scaled_dot_product_attention(qr, repeat_kv(k, heads // kv), repeat_kv(v, heads // kv),
                                         is_causal=True, scale=scale).transpose(1, 2)
    (ref * w).sum().backward()

    module = types.SimpleNamespace(num_key_value_groups=heads // kv)
    qh = q.to(device, torch.bfloat16).requires_grad_()
    out, _ = hpu_attention.fused_attention_forward(
        module, qh, k.to(device, torch.bfloat16), v.to(device, torch.bfloat16), None, scaling=scale)
    (out.float() * w.to(device)).sum().backward()
    htcore.mark_step()
    return ((out.float().cpu() - ref.detach()).abs().max().item(),
            (qh.grad.float().cpu() - qr.grad).abs().max().item())


def main() -> int:
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("hpu")
    ok = reduce_ok(device, rank, world)
    speed = tflops(device)
    out_err, grad_err = attention_error(device)
    # bf16 against fp32: outputs are within a few hundredths; a wrong argument
    # order (no causal mask, wrong scale) is off by whole units.
    good = ok and out_err < 0.05 and grad_err < 0.5
    print(f"[rank {rank}/{world}] all-reduce {'ok' if ok else 'WRONG'};"
          f" bf16 matmul {speed:.0f} TFLOPS;"
          f" FusedSDPA vs reference: output {out_err:.4f}, dq {grad_err:.4f}"
          f" -> {'ok' if good else 'NOT OK'}", flush=True)
    dist.destroy_process_group()
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
