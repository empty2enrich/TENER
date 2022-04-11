# -*- encoding: utf-8 -*-
#
# Author: LL
# global pointer
#
# cython: language_level=3





import torch
# from curses import noecho
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from .utils import f1_globalpointer, global_pointer_loss, sparse_global_loss, f1_globalpointer_sparse
from modules.transformer import TransformerEncoder

from torch import nn
import torch.nn.functional as F
from my_py_toolkit.torch.transformers_pkg import load_bert
from .models import EfficientGlobalPointer



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


    def _forward(self, inputs_idx, segments, entity_label, head_label, tail_label, sparse=False, bigrams=None):
        mask_tensor = inputs_idx.ne(0).to(int)
        chars, _ = self.bert(inputs_idx, mask_tensor)
        # chars = self.embed(chars)
        # if self.bi_embed is not None:
        #     bigrams = self.bi_embed(bigrams)
        #     chars = torch.cat([chars, bigrams], dim=-1)

        # chars = self.in_fc(chars)
        
        entity = self.entity(chars, mask_tensor)
        head = self.head(chars, mask_tensor)
        tail = self.tail(chars, mask_tensor)
        # out = torch.cat([entity, head, tail], dim=1)
        if entity_label is None:
            return (entity, head, tail)
        else:
            if sparse:
                loss_sum = 0
                f1_sum = 0
                res = {}
                for name, y_pre, y_true in zip(['entity', 'head', 'tail'], [entity, head, tail], [entity_label, head_label, tail_label]):
                    loss = sparse_global_loss(y_true, y_pre)
                    # f1 = f1_globalpointer_sparse(y_true, y_pre)
                    res[f'{name}_loss'] = loss
                    # res[f'{name}_f1'] = f1
                    loss_sum += loss
                    # f1_sum += f1
                res['loss'] = loss_sum / 3
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

    def forward(self, chars, segments, entity=None, head=None, tail=None, sparse=True, bigrams=None):
        return self._forward(chars, segments, entity, head, tail, sparse, bigrams)

    def predict(self, chars, segments, sparse=True, bigrams=None):
        return self._forward(chars, segments, sparse=sparse, bigrams=bigrams)

from transformers.modeling_bert import BertPreTrainedModel, BertModel
class GPLinker(nn.Module):
    def __init__(self, config, predicate2id, head_size=64, use_efficient=False):
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
        self.bert =  BertModel.from_pretrained(config) #load_bert(config, True)
        
        self.nums_tag = len(predicate2id)
        # todo: 试试不用效果
        # self.in_fc = nn.Linear(embed_size, d_model)
        d_model = 768
        self.entity = EfficientGlobalPointer(2, head_size, d_model)
        # 测试下使用与不使用 rope, tril 效果
        self.head = EfficientGlobalPointer(len(predicate2id), head_size, d_model, False, False)
        self.tail = EfficientGlobalPointer(len(predicate2id), head_size, d_model, False, False)


    def _forward(self, inputs_idx, segments, entity_label, head_label, tail_label, sparse=False, bigrams=None):
        mask_tensor = inputs_idx.ne(0).to(int)
        chars, _ = self.bert(inputs_idx, mask_tensor)
        # chars = self.embed(chars)
        # if self.bi_embed is not None:
        #     bigrams = self.bi_embed(bigrams)
        #     chars = torch.cat([chars, bigrams], dim=-1)

        # chars = self.in_fc(chars)
        
        entity = self.entity(chars, mask_tensor)
        head = self.head(chars, mask_tensor)
        tail = self.tail(chars, mask_tensor)
        # out = torch.cat([entity, head, tail], dim=1)
        if entity_label is None:
            return (entity, head, tail)
        else:
            if sparse:
                loss_sum = 0
                f1_sum = 0
                res = {}
                for name, y_pre, y_true in zip(['entity', 'head', 'tail'], [entity, head, tail], [entity_label, head_label, tail_label]):
                    loss = sparse_global_loss(y_true, y_pre)
                    # f1 = f1_globalpointer_sparse(y_true, y_pre)
                    res[f'{name}_loss'] = loss
                    # res[f'{name}_f1'] = f1
                    loss_sum += loss
                    # f1_sum += f1
                res['loss'] = loss_sum / 3
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

    def forward(self, chars, segments, entity=None, head=None, tail=None, sparse=True, bigrams=None):
        return self._forward(chars, segments, entity, head, tail, sparse, bigrams)

    def predict(self, chars, segments, sparse=True, bigrams=None):