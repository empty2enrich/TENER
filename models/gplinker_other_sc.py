# -*- encoding: utf-8 -*-
#
# Author: LL
#
# 别人实现的源码
#
# cython: language_level=3

import numpy as np
import torch

from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel

parent_cls, base_cls = BertPreTrainedModel, BertModel


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 *
                               head_size, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            0, inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            0, output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]],
                              axis=-1).reshape_as(qw)

            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]],
                              axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] *
                attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(head_size * 2, heads * 2, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]],
                              axis=-1).reshape_as(qw)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]],
                              axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] *
                attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits


def sparse_multilabel_categorical_crossentropy(
    y_true, y_pred, mask_zero=False, epsilon=1e-7, Inf=1e12
):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + Inf
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_loss(y_pred, y_true):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(
        y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()


class AutoModelGPLinker(parent_cls):
    def __init__(self, config, predicate2id, head_size=64, use_efficient=False):
        super().__init__(config)
        # if exist_add_pooler_layer:
        #     setattr(
        #         self,
        #         self.base_model_prefix,
        #         base_cls(config, add_pooling_layer=False),
        #     )
        # else:
        setattr(self, self.base_model_prefix, base_cls(config))
        if use_efficient:
            gpcls = EfficientGlobalPointer
        else:
            gpcls = GlobalPointer
        self.entity_output = gpcls(
            hidden_size=config.hidden_size, heads=2, head_size=head_size
        )
        self.head_output = gpcls(
            hidden_size=config.hidden_size,
            heads=len(predicate2id),
            head_size=head_size,
            RoPE=False,
            tril_mask=False,
        )
        self.tail_output = gpcls(
            hidden_size=config.hidden_size,
            heads=len(predicate2id),
            head_size=head_size,
            RoPE=False,
            tril_mask=False,
        )
        # self.post_init()
        # self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):

        outputs = getattr(self, self.base_model_prefix)(
            input_ids=input_ids,
            attention_mask=attention_mask #,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=False,
            # **kwargs,
        )
        last_hidden_state = outputs[0]
        entity_output = self.entity_output(
            last_hidden_state, attention_mask=attention_mask
        )
        head_output = self.head_output(
            last_hidden_state, attention_mask=attention_mask
        )
        tail_output = self.tail_output(
            last_hidden_state, attention_mask=attention_mask
        )

        spo_output = (entity_output, head_output, tail_output)
        loss = None
        if labels is not None:
            loss = (
                sum(
                    [
                        globalpointer_loss(o, l)
                        for o, l in zip(spo_output, labels)
                    ]
                )
                / 3
            )
        output = (spo_output,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output



