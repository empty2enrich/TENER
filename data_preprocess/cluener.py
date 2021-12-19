# -*- encoding: utf-8 -*-
#
# Author: LL
#
# 预处理数据
#
# cython: language_level=3

import collections
import json
import os
import torch

from my_py_toolkit.file.file_toolkit import read_file, readjson, writejson
from my_py_toolkit.ml.data.text_preprocess import tokenize_chinese
from my_py_toolkit.torch.transformers_pkg import bert_tokenize, load_bert
from my_py_toolkit.torch.utils import get_k_folder
from my_py_toolkit.decorator.decorator import fn_timer

from torch.utils.data import DataLoader, TensorDataset

# 处理 label
@fn_timer()
def handle_labels(labels):
    """
    处理 lables, 方便后续使用。

    处理结果：每行返回一个 dict, key 为 label name, value 当前 label 的 idx 范围，[(start, end)].

    Args:
        labels ([dict]): 数据 label
    Return:
        res(list(dict)): [{label: [(start, end)]}], 不包含 end.
    """
    res = []
    for label_line in labels:
        cur = {}
        for name_label, detail_label in label_line.items():
            scopes_cur_label = []
            for _, scopes in detail_label.items():
                scopes_cur_label.extend([(s[0], s[1] + 1) for s in scopes])
            cur[name_label] = scopes_cur_label
        res.append(cur)
    return res


# 处理 token, labels, 返回每个 token 的 label。


def check_token_label(label_ori, idx_transfer):
    """
    检查 token 的 label 是否正确。（主要看 tokenize 后label 与 toekn 是否一一对应， tokenize 有 token 可能包含多个字符。）

    1、根据 label 的起始位置找到 token 的起始位置
    2、根据找到的 token 起始位置找到对应的 label 起始位置 label_token
    3、比较 label 与 token label 是否一致。

    Args:
        label_ori (list(int)): label 在原始文本中标注的 idx 范围（start, end）, 不包含 end.
        idx_transfer:
        # new2ori_idx (list((start, end))): token 与原文本的对应关系, 不包含 end.
        # ori2new_idx (list(int)): 原文本与 token 的对应关系。
    """
    start, end = label_ori
    start_token, end_token =idx_transfer.to_new(start), idx_transfer.to_new(end)
    if start == idx_transfer.to_ori(start_token) and end == idx_transfer.to_ori(end_token):
        return True
    else:
        return False

  
def generate_label(name_label, nums_label, split=False):
    """
    生成 label.

    Args:
        name_label (str): 标签名。
        nums_label (int): 标签个数。
        split (bool, optional): 是否拆分标签. Defaults to False. 若为 true, 根据标签个数加上前缀 'B-'(前)、'I-'(中)、'E-'(后)
    """
    assert nums_label > 0
    
    name_label = name_label.upper()
    if not split:
        return [name_label] * nums_label
    else:
        label_gene = []
        if nums_label == 1:
            label_gene.append("S-" + name_label)
        else:
            for i in range(nums_label):
                if i==0:
                    label_gene.append("B-" + name_label)
                elif i > 0 and i < nums_label - 1:
                    label_gene.append("I-" + name_label)
                else:
                    label_gene.append("E-" + name_label)
        return label_gene

@fn_timer()
def conv2tok_lab_pair(source, labels, tokenizer, split_label=False):
    """
    将数据转换为 (token, label) 对。

    Args:
        source (list(str)): 原文本
        labels (list(dict{label: [(start, end)]})): 原数据 label。scope 包含 end。
        tokenizer (Tokenizer): 对文本进行 tokenize 操作的类。
        split_label(bool): 是否拆分 label, 如: name 拆为: B-NAME, I-NAME, E-NAME.
    
    Return:
        (list((tokens, label))): 返回 source 中每个 txt 的 token 与 label 对。
    """
    pairs = []
    for txt, label_detail in zip(source, labels):
        tokens, idx_tranfer = tokenize_chinese(tokenizer, txt, True)
        labels_tokens = ["O"] * len(tokens)
        is_valid = True
        for name_label, scopes in label_detail.items():
            if not all([check_token_label(s, idx_tranfer) for s in scopes]):
                is_valid = False
                break
            for start, end in scopes:
                start_token, end_token = idx_tranfer.to_new(start), idx_tranfer.to_new(end)
                labels_tokens[start_token:end_token] = generate_label(name_label, end_token - start_token, split_label)
        if not is_valid:
            print(f"Label error: ==================\ntxt: {txt}, label：{label_detail}")
        else:
            pairs.append((tokens, labels_tokens))
    return pairs
                
#
@fn_timer() 
def convert_features(pairs, max_len, tags_mapping, tokenizer):
    """
    转换为模型输入。

    Args:
        pairs (list(tuple)): [description]
        max_len (int): [description]
        tags_mapping (dict): [description]
        tokenizer (Tokenizer): [description]

    Returns:
        tuple: inputs_idx, labels_idx, segments, mask
    """
    inputs_idx = []
    segments = []
    labels_idx = []
    mask = []
    
    for tokens, labels in pairs:
        tokens = ['[CLS]'] + tokens[:max_len - 2] + ['[SEP]']
        labels = ['O'] + labels[:max_len - 2] + ['O']
        segment = [0] * max_len
        cur_mask = [1] * len(tokens)
        
        tokens +=  ['[PAD]'] * (max_len - len(tokens))
        labels +=  ['O'] * (max_len - len(labels))
        cur_mask += [0] * (max_len - len(cur_mask))
        
        inputs_idx.append(tokenizer.convert_tokens_to_ids(tokens))
        segments.append(segment)
        labels_idx.append([tags_mapping[label] for label in labels])
        mask.append(cur_mask)
    
    return inputs_idx, labels_idx, segments, mask
        
@fn_timer()        
def get_k_folder_dataloder(data_paths, bert_cfg, tags_path, max_len, split_label,
                         batch_size, cache_dir):
    """
    返回 k 折校验每次返回的数据。

    Args:
        data_paths ([type]): [description]
        bert_cfg ([type]): [description]
        tags_path ([type]): [description]
        max_len ([type]): [description]
        split_label ([type]): [description]
        batch_size ([type]): [description]
        cache_dir ([type]): [description]

    Yields:
        [type]: [description]
    """
    pairs = None
    tokenizer = bert_tokenize(bert_cfg)
    inputs_idx, labels_idx, segments, mask = None, None, None, None
    
    tags_mapping = {label:i for i, label in enumerate(readjson(tags_path))}
    
    # 得到 token, label 对。
    if os.path.exists(os.path.join(cache_dir, 'pairs.json')):
        pairs = readjson(os.path.join(cache_dir, 'pairs.json'))
    else:
        source, labels = [], []
        datas = []
        for p in data_paths:
           datas.extend(read_file(p, '\n'))
        datas = [d for d in datas if d]
        for line in datas:
            cur = json.loads(line)
            source.append(cur['text'])    
            labels.append(cur['label'])
        pairs = conv2tok_lab_pair(source, handle_labels(labels), tokenizer, split_label)
        writejson(pairs, os.path.join(cache_dir, 'pairs.json'))
    
    # 得到模型输入数据
    if os.path.exists(os.path.join(cache_dir, 'inputs_idx.json')):
        inputs_idx = readjson(os.path.join(cache_dir, 'inputs_idx.json'))    
        labels_idx = readjson(os.path.join(cache_dir, 'labels_idx.json'))
        segments = readjson(os.path.join(cache_dir, 'segments.json'))
        mask = readjson(os.path.join(cache_dir, 'mask.json'))
    else:
        inputs_idx, labels_idx, segments, mask = convert_features(pairs, max_len, tags_mapping, tokenizer)
        writejson(inputs_idx, os.path.join(cache_dir, 'inputs_idx.json'))
        writejson(labels_idx, os.path.join(cache_dir, 'labels_idx.json'))
        writejson(segments, os.path.join(cache_dir, 'segments.json'))
        writejson(mask, os.path.join(cache_dir, 'mask.json'))
    
    for train_d, test_d in get_k_folder(inputs_idx, labels_idx, segments, mask, k=5):
        train_dataloader = DataLoader(TensorDataset(*[torch.tensor(d, dtype=torch.long) for d in train_d]), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(TensorDataset(*[torch.tensor(d, dtype=torch.long) for d in test_d]), batch_size=batch_size, shuffle=True)
        yield train_dataloader, test_dataloader
        


