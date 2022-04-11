# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3

from tkinter.messagebox import NO
import torch
import torch.nn.functional as F
from my_py_toolkit.torch.transformer_utils import gen_pos_emb
from my_py_toolkit.torch.tensor_toolkit import mask

from .utils import rope

# efficient globalpointer
class EfficientGlobalPointer(torch.nn.Module):
    def __init__(self, num_head, head_size, dim, use_rope=True, mask_tril=True):
        super().__init__()
        self.use_rope = use_rope
        self.mask_tril = mask_tril
        self.head_size = head_size
        self.linear_1 = torch.nn.Linear(dim, head_size * 2)
        self.linear_2 = torch.nn.Linear(head_size * 2, num_head * 2)

    def forward(self, inputs, mask_t):
        inputs = self.linear_1(inputs)
        q, k = inputs[..., ::2], inputs[..., 1::2]
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = q@k.transpose(-1, -2) / self.head_size ** 0.5
        bias = self.linear_2(inputs).transpose(-1, -2) / 2
        logits = att.unsqueeze(1) + bias[:, ::2, :].unsqueeze(-1) + bias[:, 1::2, :].unsqueeze(-2)
        if mask_t is not None:
            att_mask = 1 - mask_t[:, None, None, :] * mask_t[:, None, :, None]
            logits -= att_mask * 1e12
        # logits = mask(logits, mask_t, -1, -10000)
        # logits = mask(logits, mask_t, -2, -10000)
        logits[:, :, 0, :] = -10000
        logits[:, :, :, 0] = -10000
        if self.mask_tril:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits -= mask * 1e12
            # logits = logits.masked_fill(torch.tril(torch.ones_like(logits).to(bool), -1), -10000)
        return logits



class EfficientGlobalPointer(torch.nn.Module):
    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, use_bias=True, tril_mask=True):
        super().__init__()
        self.use_rope = RoPE
        self.mask_tril = tril_mask
        self.head_size = head_size
        self.linear_1 = torch.nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.linear_2 = torch.nn.Linear(head_size * 2, heads * 2, bias=use_bias)

    def forward(self, inputs, attention_mask=None):
        inputs = self.linear_1(inputs)
        q, k = inputs[..., ::2], inputs[..., 1::2]
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = q@k.transpose(-1, -2) / self.head_size ** 0.5
        bias = self.linear_2(inputs).transpose(-1, -2) / 2
        logits = att.unsqueeze(1) + bias[:, ::2, :].unsqueeze(-1) + bias[:, 1::2, :].unsqueeze(-2)
        if attention_mask is not None:
            att_mask = 1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            logits -= att_mask * 1e12
        # logits = mask(logits, mask_t, -1, -10000)
        # logits = mask(logits, mask_t, -2, -10000)
        logits[:, :, 0, :] = -10000
        logits[:, :, :, 0] = -10000
        if self.mask_tril:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits -= mask * 1e12
            # logits = logits.masked_fill(torch.tril(torch.ones_like(logits).to(bool), -1), -10000)
        return logits
