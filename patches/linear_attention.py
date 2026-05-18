# Copyright (c) 2026 LSAIE Course Project. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class LinearAttentionCore(nn.Module):
    """O(T) replacement for DotProductAttention using the kernel trick.

    Replaces softmax(QK^T/sqrt(d))V with phi(Q)(phi(K)^T V) where phi(x)=elu(x)+1.
    Non-causal (bidirectional) — suited for throughput benchmarking.

    Inputs/outputs follow DotProductAttention:
      query/key/value: [seq, batch, heads_per_partition, head_dim]
      returns:         [seq, batch, hidden_per_partition]
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

        q = torch.nn.functional.elu(query) + 1
        k = torch.nn.functional.elu(key) + 1

        kv = torch.einsum('tbhd,tbhe->bhde', k, value)
        z = k.sum(dim=0)

        numerator = torch.einsum('tbhd,bhde->tbhe', q, kv)
        denominator = (
            torch.einsum('tbhd,bhd->tbh', q, z)
            .unsqueeze(-1)
            .clamp(min=1e-6)
        )

        return (numerator / denominator).reshape(sq, b, np_heads * hn)
