# -*- encoding: utf-8 -*-
#
# Author: LL
# global pointer
#
# 别人实现的 gplinker
# cython: language_level=3





import torch
# from curses import noecho
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from .utils import f1_globalpointer, global_pointer_loss, sparse_global_loss, f1_globalpointer_sparse
from modules.transformer import TransformerEncoder

from torch import nn
import torch.nn.functional as F
from my_py_toolkit.torch.transformers_pkg import load_bert
# from .models import EfficientGlobalPointer


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        heads=12,
        head_size=64,
        hidden_size=768,
        RoPE=True,
        tril_mask=True,
        use_bias=True,
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

class GPLinker(nn.Module):
    def __init__(self, tag_vocab, bert_cfg, dim_embedding, d_model, gp_head_size=64):
        """

        :param tag_vocab: fastNLP Vocabulary, ner 的 label
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()

        # self.embed = embed
        self.bert = load_bert(bert_cfg, True)
        embed_size = dim_embedding
        self.nums_tag = len(tag_vocab)
        # todo: 试试不用效果
        # self.in_fc = nn.Linear(embed_size, d_model)
        d_model = embed_size
        self.entity = EfficientGlobalPointer(2, gp_head_size, d_model)
        # 测试下使用与不使用 rope, tril 效果
        self.head = EfficientGlobalPointer(len(tag_vocab), gp_head_size, d_model, False, False)
        self.tail = EfficientGlobalPointer(len(tag_vocab), gp_head_size, d_model, False, False)


    def _forward(self, input_ids=None, mask_tensor=None, target=None, sparse=False, bigrams=None):
        # mask_tensor = input_ids.ne(0).to(int)
        chars, _ = self.bert(input_ids, mask_tensor)
        # chars = self.embed(chars)
        # if self.bi_embed is not None:
        #     bigrams = self.bi_embed(bigrams)
        #     chars = torch.cat([chars, bigrams], dim=-1)

        # chars = self.in_fc(chars)
        
        entity = self.entity(chars, mask_tensor)
        head = self.head(chars, mask_tensor)
        tail = self.tail(chars, mask_tensor)
        # out = torch.cat([entity, head, tail], dim=1)
        if target is None:
            return (entity, head, tail)
        else:
            if sparse:
                loss_sum = 0
                f1_sum = 0
                res = {}
                for name, y_pre, y_true in zip(['entity', 'head', 'tail'], [entity, head, tail], target):
                    loss = sparse_global_loss(y_true, y_pre)
                    # f1 = f1_globalpointer_sparse(y_true, y_pre)
                    res[f'{name}_loss'] = loss
                    # res[f'{name}_f1'] = f1
                    loss_sum += loss
                    # f1_sum += f1
                res['loss'] = loss_sum
                # res['f1'] = f1_sum / 3
                return res
            else:
                pass
                # loss = global_pointer_loss(target, out)
                # f1 = f1_globalpointer(target, out)
                # return {'loss': loss, 'f1': f1}
            
        
        # logits = F.log_softmax(chars, dim=-1)
        # if target is None:
        #     paths, _ = self.crf.viterbi_decode(logits, mask)
        #     return {'pred': paths}
        # else:
        #     loss = self.crf(logits, target, mask)
        #     return {'loss': loss}

    def forward(self, chars, mask_tensor, label=None, sparse=True, bigrams=None):
        return self._forward(chars, mask_tensor, label, sparse, bigrams)

    def predict(self, chars, mask_tensor=None, sparse=True, bigrams=None):
        return self._forward(chars, mask_tensor, sparse=sparse, bigrams=bigrams)

# class AutoModelGPLinker(parent_cls):
#     def __init__(self, config, predicate2id, head_size=64, use_efficient=False):
#         super().__init__(config)
#         if exist_add_pooler_layer:
#             setattr(
#                 self,
#                 self.base_model_prefix,
#                 base_cls(config, add_pooling_layer=False),
#             )
#         else:
#             setattr(self, self.base_model_prefix, base_cls(config))
#         if use_efficient:
#             gpcls = EfficientGlobalPointer
#         else:
#             gpcls = GlobalPointer
#         self.entity_output = gpcls(
#             hidden_size=config.hidden_size, heads=2, head_size=head_size
#         )
#         self.head_output = gpcls(
#             hidden_size=config.hidden_size,
#             heads=len(predicate2id),
#             head_size=head_size,
#             RoPE=False,
#             tril_mask=False,
#         )
#         self.tail_output = gpcls(
#             hidden_size=config.hidden_size,
#             heads=len(predicate2id),
#             head_size=head_size,
#             RoPE=False,
#             tril_mask=False,
#         )
#         self.post_init()

#     def forward(
#         self,
#         input_ids,
#         attention_mask=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         **kwargs
#     ):

#         outputs = getattr(self, self.base_model_prefix)(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=False,
#             **kwargs,
#         )
#         last_hidden_state = outputs[0]
#         entity_output = self.entity_output(
#             last_hidden_state, attention_mask=attention_mask
#         )
#         head_output = self.head_output(
#             last_hidden_state, attention_mask=attention_mask
#         )
#         tail_output = self.tail_output(
#             last_hidden_state, attention_mask=attention_mask
#         )

#         spo_output = (entity_output, head_output, tail_output)
#         loss = None
#         if labels is not None:
#             loss = (
#                 sum(
#                     [
#                         globalpointer_loss(o, l)
#                         for o, l in zip(spo_output, labels)
#                     ]
#                 )
#                 / 3
#             )
#         output = (spo_output,) + outputs[1:]
#         return ((loss,) + output) if loss is not None else output
