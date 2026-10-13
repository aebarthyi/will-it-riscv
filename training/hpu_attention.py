"""Gaudi's fused attention as a transformers attention implementation.

SDPA's math kernel holds every score, heads x length^2: at 32k tokens that is
64 GB a layer for Qwen3-4B, past even a Gaudi 2's 96. Habana's FusedSDPA in
its recompute mode works the way flash attention does, a tile at a time, and
recomputes the scores in the backward instead of keeping them.

    import hpu_attention
    hpu_attention.register()
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="fused")

Keys and values are repeated to the query heads first (Qwen3's grouped
queries), as SDPA's own path does; that costs length x heads x 128 per layer,
not length^2.
"""

from __future__ import annotations

import torch
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

NAME = "fused"


def fused_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert dropout == 0.0, "fused attention here does not do dropout"
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    # Training sequences carry no padding mask (pads only follow the text), so
    # the mask is None and causal does the masking.
    causal = attention_mask is None
    # (query, key, value, mask, dropout, is_causal, scale, softmax mode, recompute)
    out = FusedSDPA.apply(query, key, value, attention_mask, 0.0, causal, scaling, "None", True)
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    """Make ``attn_implementation="fused"`` available."""
    AttentionInterface.register(NAME, fused_attention_forward)
    # The same mask SDPA gets: none at all, when the batch has no padding.
    AttentionMaskInterface.register(NAME, sdpa_mask)
