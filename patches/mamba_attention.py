# Copyright (c) 2026 LSAIE Course Project. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


def _parallel_scan(decay: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Causal linear recurrence h_t = decay_t * h_{t-1} + inp_t, h_0=0.

    Log-cumsum trick: no Python loop, fully parallel on GPU.
    decay: [T, B, H, D] in (0,1)   inp: [T, B, H, D]
    """
    log_d = torch.log(decay.clamp(min=1e-8))
    cum_log_d = torch.cumsum(log_d, dim=0)
    return torch.exp(cum_log_d) * torch.cumsum(torch.exp(-cum_log_d) * inp, dim=0)


class MambaCore(nn.Module):
    """Selective SSM (Mamba-inspired) as core_attention replacement.

    O(T) causal recurrence via parallel scan, replacing O(T^2) attention.

    Mapping from attention interface:
      V  -> SSM input x_t
      Q  -> timescale Delta_t = softplus(Q)/sqrt(D), and output gate
      K  -> selective input gate B_t
    Fixed A = -1 (exponential decay per feature; no learnable state matrix).

    Recurrence:
      a_t = exp(-Delta_t)            decay in (0, 1)
      b_t = Delta_t * K_t * V_t      input contribution
      h_t = a_t * h_{t-1} + b_t     (parallel scan, O(T))
      y_t = Q_t * h_t                selective read
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        softmax_scale: Optional[float] = None,
        pg_collection=None,
    ):
        super().__init__()
        self.config = config

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        /,
        *,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        sq, b, np_heads, hn = query.shape

        if key.shape[2] < np_heads:
            repeat = np_heads // key.shape[2]
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        dt = F.softplus(query) * (hn ** -0.5)
        decay = torch.exp(-dt)
        inp = dt * key * value

        h = _parallel_scan(decay, inp)
        return (query * h).reshape(sq, b, np_heads * hn)
