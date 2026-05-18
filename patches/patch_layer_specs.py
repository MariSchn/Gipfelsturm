#!/usr/bin/env python3
"""Inject LinearAttentionCore/_CORE_ATTN into gpt_layer_specs.py.

Run from the Megatron-LM root directory after git apply patches/*.patch.
"""
import os, sys

path = 'megatron/core/models/gpt/gpt_layer_specs.py'
if not os.path.exists(path):
    sys.exit(f'ERROR: {path} not found – run from Megatron-LM root')

text = open(path).read()

if 'LinearAttentionCore' in text:
    print(f'{path}: already patched, skipping')
    sys.exit(0)

old_import = 'from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules'
if old_import not in text:
    sys.exit(f'ERROR: expected import line not found in {path}')

addition = (
    '\nimport os as _os'
    '\nfrom megatron.core.transformer.linear_attention import LinearAttentionCore'
    '\nfrom megatron.core.transformer.mamba_attention import MambaCore'
    '\nfrom megatron.core.transformer.xlstm_attention import XLSTMCore'
    '\n_CORE_ATTN = {'
    '\n    "linear": LinearAttentionCore,'
    '\n    "mamba":  MambaCore,'
    '\n    "xlstm":  XLSTMCore,'
    '\n}.get(_os.environ.get("ATTN_BACKEND", "linear"), LinearAttentionCore)'
)
text = text.replace(old_import, old_import + addition, 1)

core_attn_old = 'core_attention=backend.core_attention()'
if core_attn_old not in text:
    sys.exit(f'ERROR: core_attention pattern not found in {path}')
text = text.replace(core_attn_old, 'core_attention=_CORE_ATTN', 1)

open(path, 'w').write(text)
print(f'Patched {path}')
