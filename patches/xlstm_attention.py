# Copyright (c) 2026 LSAIE Course Project. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mamba_attention import _parallel_scan


class XLSTMCore(nn.Module):
    """xLSTM (sLSTM-style) as core_attention replacement.

    Key xLSTM innovation: separate forget/input/output gates with causal state.
    Uses parallel scan for O(T) compute (vs O(T^2) for softmax attention).

    Mapping from attention interface:
      V -> cell input
      Q -> forget gate f = sigmoid(Q), output gate o = sigmoid(Q + K)
      K -> input gate  i = sigmoid(K)

    Recurrence:
      c_t = f_t * c_{t-1} + i_t * V_t     cell state  (parallel scan)
      n_t = f_t * n_{t-1} + i_t           normaliser  (parallel scan)
      y_t = o_t * c_t / max(n_t, 1)
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

        f = torch.sigmoid(query)
        i = torch.sigmoid(key)
        o = torch.sigmoid(query + key)

        c = _parallel_scan(f, i * value)
        n = _parallel_scan(f, i)

        y = o * c / n.clamp(min=1.0)
        return y.reshape(sq, b, np_heads * hn)
