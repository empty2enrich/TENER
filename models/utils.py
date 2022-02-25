# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3

# sin 位置编码

import torch
from my_py_toolkit.torch.transformer_utils import gen_pos_emb
from my_py_toolkit.torch.tensor_toolkit import mask

################################ toolkit #######################################

def rope(inputs):
    """rotary position embedding"""
    length, dim = inputs.shape[-2:]
    pos = gen_pos_emb(length, dim)
    inputs_2 = torch.zeros_like(inputs)
    inputs_2[..., ::2] = -inputs[..., 1::2]
    inputs_2[..., 1::2] = inputs[..., ::2]
    sin, cos = torch.zeros_like(pos), torch.zeros_like(pos)
    sin[:, ::2], sin[:, 1::2] = pos[:, ::2], pos[:, ::2]
    cos[:, ::2], cos[:, 1::2] = pos[:, 1::2], pos[:, 1::2]
    return inputs * cos + inputs_2 * sin

def multilabel_categorical_crossentropy(y_true, y_pre):
    y_pre = (1 - 2* y_true) * y_pre
    y_neg = mask(y_pre, y_true, -1, -10000)
    y_pos = mask(y_pre, 1-y_true, -1, -10000)
    zero = torch.zeros_like(y_true[..., :1])
    y_neg = torch.cat([y_neg, zero], -1)
    y_pos = torch.cat([y_pos, zero], -1)
    return torch.logsumexp(y_neg, -1) + torch.logsumexp(y_pos, -1)


def global_pointer_loss(y_true, y_pre):
    b, h = y_pre.shape[:2]
    y_pre = y_pre.reshape(b*h, -1)
    y_true = y_true.permute(0, 3, 1, 2).reshape(b*h, -1)
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pre))


def f1_globalpointer(y_true, y_pre):
    y_pre = y_pre.greater(0)
    return 2 * (y_true * y_pre).sum() / (y_true.sum() + y_pre.sum())


def em_globalpoiter(y_true, y_pre):
    y_pre = y_pre.greater(0)
    return (y_true * y_pre).sum()/ y_pre.sum()    

def predict_globalpointer(y_pre, tags):
    res = [{}] * y_pre.shape[0]
    idx = (y_pre>0).nonzero()
    for b, tag_idx, start, end in idx.tolist():
        if tags[tag_idx] not in res[b]:
            res[b][tags[tag_idx]] = []
        res[b][tags[tag_idx]].append((start, end))
    return res

def predict_globalpointer_compare(y_pre, tags):
    res = [[] for _ in range(y_pre.shape[0])]
    idx = (y_pre>0).nonzero()
    for b, tag_idx, start, end in idx.tolist():
        res[b].append((tags[tag_idx], (start, end)))
        
    return res


def compare_res(y_pre, y_true, tags, txts, new2ori_idxs):
    y_pre = predict_globalpointer_compare(y_pre, tags)
    y_true = predict_globalpointer_compare(y_true, tags)
    res = []
    for y_pre_cur, y_true_cur, txt, new2ori_idx in zip(y_pre, y_true, txts, new2ori_idxs):
        y_pre_cur, y_true_cur = set(y_pre_cur), set(y_true_cur)
        cur_res = {}
        pre_res, true_res = [], []
        for tag, (start, end) in y_pre_cur - y_true_cur:
            # 减一原因：第一个 token 为 [CLS]
            end = min(end - 1, len(new2ori_idx) - 1)
            start_txt, end_txt = new2ori_idx[start - 1][0], new2ori_idx[end][1]
            ner_txt = txt[start_txt:end_txt]
            pre_res.append((tag, ner_txt, (start, end), (start_txt, end_txt)))
        for tag, (start, end) in y_true_cur - y_pre_cur:
            end = min(end - 1, len(new2ori_idx) - 1)
            start_txt, end_txt = new2ori_idx[start - 1][0], new2ori_idx[end - 1][1]
            ner_txt = txt[start_txt:end_txt]
            true_res.append((tag, ner_txt, (start, end), (start_txt, end_txt)))
        if pre_res:
            cur_res['pre'] = pre_res
        if true_res:
            cur_res['true'] = pre_res
        if cur_res:
            cur_res['txt'] = txt
        if cur_res:
            res.append(cur_res)
    return res
    
