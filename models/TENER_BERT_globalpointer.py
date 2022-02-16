# -*- encoding: utf-8 -*-
#
# Author: LL
# global pointer
#
# cython: language_level=3





import torch
from curses import noecho
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from models.utils import f1_globalpointer, global_pointer_loss
from modules.transformer import TransformerEncoder

from torch import nn
import torch.nn.functional as F
from my_py_toolkit.torch.transformers_pkg import load_bert
from .models import EfficientGlobalPointer
# todo: 测试用
# from transformers.optimization import AdamW
# from torch.optim import AdamW, Adam



class TENER(nn.Module):
    def __init__(self, tag_vocab, bert_cfg, dim_embedding, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  
                 # bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
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
        # self.bi_embed = None
        # if bi_embed is not None:
        #     self.bi_embed = bi_embed
        #     embed_size += self.bi_embed.embed_size

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))

       
        self.global_pointer = EfficientGlobalPointer(len(tag_vocab), d_model)
        # trans = allowed_transitions(tag_vocab, include_start_end=True)
        # self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)


    def _forward(self, inputs_idx, target, segments, bigrams=None):
        mask_tensor = inputs_idx.ne(0)
        chars, _ = self.bert(inputs_idx, segments)
        # chars = self.embed(chars)
        # if self.bi_embed is not None:
        #     bigrams = self.bi_embed(bigrams)
        #     chars = torch.cat([chars, bigrams], dim=-1)

        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask_tensor)

        chars = self.fc_dropout(chars)
        
        chars = self.out_fc(chars)
        
        chars = self.global_pointer(chars, mask_tensor)
        if target is None:
            return chars
        else:
            loss = global_pointer_loss(target, chars)
            f1 = f1_globalpointer(target, chars)
            return {'loss': loss, 'f1': f1}
            
        
        # logits = F.log_softmax(chars, dim=-1)
        # if target is None:
        #     paths, _ = self.crf.viterbi_decode(logits, mask)
        #     return {'pred': paths}
        # else:
        #     loss = self.crf(logits, target, mask)
        #     return {'loss': loss}

    def forward(self, chars, target, segments, bigrams=None):
        return self._forward(chars, target, segments, bigrams)

    def predict(self, chars, segments, bigrams=None):
        return self._forward(chars, target=None, segments=segments, bigrams=bigrams)
