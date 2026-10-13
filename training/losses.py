"""The loss on the plan, without the logits of the prompt.

The plan is about 12% of a training sequence. A causal LM's forward computes
logits for every position -- at Qwen3's 151,936-token vocabulary, 12k
positions are 3.7 GB in fp16 and twice that once the loss upcasts them --
and the loss then ignores 88% of them. So the head is applied only at the
positions whose next token is supervised: the same loss, a fraction of the
memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE = -100


def causal_lm(model):
    """The ``*ForCausalLM`` under a PEFT wrapper, or the model itself."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def sliced_loss(model, input_ids: torch.Tensor, labels: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean cross-entropy over the supervised tokens, as ``model(labels=...)`` gives."""
    lm = causal_lm(model)
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False).last_hidden_state
    # Position i predicts token i + 1.
    targets = labels[:, 1:]
    keep = targets != IGNORE
    logits = lm.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), targets[keep])


def _chunk_loss(head, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(head(hidden).float().flatten(0, 1), targets.flatten(),
                           ignore_index=IGNORE, reduction="sum")


def static_loss(model, input_ids: torch.Tensor, labels: torch.Tensor,
                chunk: int = 4096) -> torch.Tensor:
    """sliced_loss's value, with shapes that follow the sequence's length alone.

    sliced_loss gathers the supervised positions, so its shapes change with
    every example, and a graph compiler (Gaudi's) compiles a graph for each.
    Here the head runs at every position, a fixed-size chunk at a time, each
    chunk checkpointed so memory holds one chunk's logits. With lengths padded
    to a few buckets, that is a few graphs.
    """
    lm = causal_lm(model)
    hidden = lm.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.shape[1], chunk):
        total = total + checkpoint(_chunk_loss, lm.lm_head, hidden[:, start:start + chunk],
                                   targets[:, start:start + chunk], use_reentrant=False)
    return total / (targets != IGNORE).sum()
