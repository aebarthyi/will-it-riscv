"""Attention in query blocks, for a card with no memory-efficient kernel.

gfx906 runs SDPA's math kernel only: it materialises every score,
heads x length^2, and holds them for the backward. Here the queries go in
blocks; each block sees the keys up to its own end (causal), and each is
checkpointed, so the backward recomputes one block's scores at a time.
Memory grows with block x length instead of length^2, for the price of
computing the scores twice.

    import chunked_attention
    chunked_attention.register(block=1024)
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="chunked")
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

NAME = "chunked"
BLOCK = 1024


def _block(query, key, value, mask, start: int, scaling: float):
    """One block of queries, starting at ``start``, against every key before its end."""
    stop = start + query.shape[2]
    key, value = key[:, :, :stop], value[:, :, :stop]
    if mask is not None:
        mask = mask[:, :, start:stop, :stop]
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
    # Causal within the block: query start + i sees keys up to start + i.
    rows = torch.arange(start, stop, device=query.device)[:, None]
    cols = torch.arange(stop, device=query.device)[None, :]
    return F.scaled_dot_product_attention(query, key, value, attn_mask=cols <= rows,
                                          scale=scaling)


def chunked_attention_forward(
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
    assert dropout == 0.0, "chunked attention does not do dropout"
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    if query.shape[2] != key.shape[2]:
        raise ValueError("chunked attention is for training: queries and keys the same length")
    block = getattr(module.config, "attention_block", BLOCK)
    outputs = []
    for start in range(0, query.shape[2], block):
        q = query[:, :, start:start + block]
        if torch.is_grad_enabled() and q.requires_grad:
            out = checkpoint(_block, q, key, value, attention_mask, start, scaling,
                             use_reentrant=False)
        else:
            out = _block(q, key, value, attention_mask, start, scaling)
        outputs.append(out)
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def register(block: int = BLOCK) -> None:
    """Make ``attn_implementation="chunked"`` available, with this block size."""
    global BLOCK
    BLOCK = block
    AttentionInterface.register(NAME, chunked_attention_forward)
    # The same mask SDPA gets: none at all, when the batch has no padding.
    AttentionMaskInterface.register(NAME, sdpa_mask)
