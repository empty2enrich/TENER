# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3

import torch
import torch.nn.functional as F
from my_py_toolkit.torch.transformer_utils import gen_pos_emb
from my_py_toolkit.torch.tensor_toolkit import mask

from .utils import rope

# efficient globalpointer
class EfficientGlobalPointer(torch.nn.Module):
    def __init__(self, num_head, head_size, dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(dim, head_size * 2)
        self.linear_2 = torch.nn.Linear(head_size * 2, num_head * 2)

    def forward(self, inputs, mask_t):
        inputs = self.linear_1(inputs)
        q, k = inputs[..., ::2], inputs[..., 1::2]
        q, k = rope(q), rope(k)
        att = q@k.transpose(-1, -2)
        bias = self.linear_2(inputs).transpose(-1, -2) / 2
        logits = att.unsqueeze(1) + bias[:, ::2, :].unsqueeze(-1) + bias[:, 1::2, :].unsqueeze(-2)
        logits = mask(logits, mask_t, -1, -10000)
        logits = mask(logits, mask_t, -2, -10000)
        logits[:, :, 0, :] = -10000
        logits[:, :, :, 0] = -10000
        logits = logits.masked_fill(torch.tril(torch.ones_like(logits).to(int)), -10000)
        return logits



