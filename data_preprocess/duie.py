# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3


import json
import numpy as np
import os
import random
import re
import sys
import torch

from collections import Counter, defaultdict
from my_py_toolkit.file.file_toolkit import readjson, writejson, read_file, get_file_paths
from my_py_toolkit.data_visulization.matplotlib import plot_coh, plot_coh_dict
from my_py_toolkit.torch.transformers_pkg import bert_tokenize
from my_py_toolkit.ml.data.text_preprocess import tokenize_chinese
from torch.utils.data import IterableDataset, DataLoader

from data_preprocess.cluener import FileDataset, FileDatasetPad


def readdata(paths):
    datas = []
    for p in paths:
        for line in read_file(p, '\n'):
            if line:
                datas.append(json.loads(line))
    return datas


def check_scope(scope, idx_transf, txt):
    s, e = scope
    # 全角空格
    while e < len(txt) - 1 and re.search('\s+', txt[e]):
        e += 1
    s_ori, e_ori = idx_transf.to_ori_scope(*idx_transf.to_new_scope(s, e))
    
    while e_ori > 0 and re.search('\s+', txt[e_ori - 1]):
        e_ori -= 1
    return  scope[0] == s_ori and scope[1] == e_ori

def handle_data(datas, bert_cfg, num_data=-1, dow_lower_case=False):
    all_tokens = []
    all_spo = []
    all_text = []
    all_label = []
    all_new2ori_idx = []
    all_ori2new_idx = []
    data_invalid = []

    tokenizer = bert_tokenize(bert_cfg)
    for data in datas:
        if num_data == 0:
            break
        txt = data['text']
        if dow_lower_case:
            txt = txt.lower()
        tokens, idx_transf = tokenize_chinese(tokenizer, txt, True)
        spo_list = []
        keys = ['object', 'subject', 'predicate', 'object_type', 'subject_type']
        for spo in data['spo_list']:
            obj, sub, pre, ot, st = [spo[k] for k in keys]
            obj_match, sub_match = [re.search(re.escape(reg), txt, re.I) for reg in [obj, sub]]
            if not obj_match or not sub_match:
                print(f'Label error, obj: {obj}, sub: {sub}, txt: {txt}')
                data_invalid.append((obj, sub, txt))
                continue
            obj_s, obj_e = obj_match.span()
            sub_s, sub_e = sub_match.span()
            if not all([check_scope(scope, idx_transf, txt) for scope in [[obj_s, obj_e], [sub_s, sub_e]]]):
                print(f'Tokens not match, obj:{obj}, sub:{sub}, txt:{txt}')
                data_invalid.append((obj, sub, txt))
                continue
            obj_s, obj_e, sub_s, sub_e = [transfer2new(v, txt, idx_transf.ori2new_idx, tokens) for v in [obj_s, obj_e, sub_s, sub_e]]
            sub_s, sub_e = idx_transf.to_new(sub_s), idx_transf.to_new(sub_e)
            spo_list.append([obj_s, obj_e, ','.join([pre, ot, st]), sub_s, sub_e])
        all_tokens.append(tokens)
        all_spo.append(spo_list)
        all_text.append(txt)
        all_ori2new_idx.append(idx_transf.ori2new_idx)
        all_new2ori_idx.append(idx_transf.new2ori_idx)
        all_label.append(data['spo_list'])
        num_data -= 1
    return (all_tokens, all_spo, all_text, all_label, all_ori2new_idx, all_new2ori_idx)

def transfer2new(idx, txt, ori2new, tokens):
    while 0 < idx < len(txt) and re.search('\s+', txt[idx]):
        idx -= 1
    if idx >= len(txt):
        return len(tokens)
    return ori2new[idx]


def split_save(datas, path, split_num=50000):
    nums = 0
    length = len(datas[0])
    while nums * split_num < length:
        writejson([item[nums * split_num: (nums + 1) * split_num] for item in datas], f"{path}_{nums}")
        nums += 1
    
# 读取 handled 数据
def read_data_handled(file_name, cache='./cache'):
    datas = []
    for p in get_file_paths(cache):
        if file_name in p:
            datas.append(readjson(p))
    tmp = datas[0]
    for item in datas[1:]:
        for i in range(len(item)):
            tmp[i].extend(item[i])
    return tmp

def get_valid_scope(spo_list):
    min_v, max_v = sys.maxsize, 0
    for s_o, e_o, _, s_s, e_s in spo_list:
        min_v = min(min_v, s_o, e_o, s_s, e_s)
        max_v = max(max_v, s_o, e_o, s_s, e_s)
    return min_v, max_v

def padding_seq(seqs, pad_val=[0, 0], dim=0):
    # 后续兼容多维度
    max_len = max([max([len(sub) if sub else 0 for sub in seq]) if seq else 0 for seq in seqs])
    for seq in seqs:
        for sub in seq:
            sub.extend([pad_val] * (max_len - len(sub)))

    return seqs
# 生成 features
def generate_features(datas_handled, max_len, labels_num, tags_mapping, tokenizer, features_num=1000, 
                      nums_save=10000, path_save='./cache/duie_features.json', stride=64, label_type='pre'):
    all_input_idx = []
    all_lables = []
    all_entity = []
    all_head = []
    all_tail = []
    all_segments = []
    all_tokens = []
    all_offset = []
    all_spo_offset = []
    all_spo = []
    all_text = []
    all_spo_ori = []
    all_ori2new_idx = []
    all_new2ori_idx = []
    nums_file = 0
    # (all_tokens, all_spo, all_text, all_label, all_ori2new_idx, all_new2ori_idx)
    for tokens, spo, txt, spo_ori, ori2new_idx, new2ori_idx in zip(*datas_handled):
        if features_num > -1 and features_num <= 0:
            break
        features_num -= 1
        offset = 0
        while offset == 0 or offset - stride + max_len - 2 < len(tokens):
            entity = [set() for _ in range(2)]
            labels_head = [set() for _ in range(labels_num)]
            labels_tail = [set() for _ in range(labels_num)]
            cur_tokens = tokens[offset: max_len - 2]
            cur_tokens = ['[CLS]'] + cur_tokens + ['[SEP]'] 
            input_idx = tokenizer.convert_tokens_to_ids(cur_tokens)
            segments = [0] * len(input_idx)
            labels = []
            cur_spo_offset = []
            for s_o, e_o, p, s_s, e_s in spo:
                if label_type == 'pre':
                    p = p.split(',')[0]
                s_o, e_o, s_s, e_s = [v - offset + 1 for v in [s_o, e_o, s_s, e_s]]
                if not all([0 < v < len(input_idx) - 1 for v in [s_o, e_o, s_s, e_s]]):
                    continue
                cur_spo_offset.append((s_o, e_o, p, s_s, e_s))
                entity[0].add((s_o, e_o - 1))
                entity[1].add((s_s, e_s - 1))
                labels_head[tags_mapping[p]].add((s_o, s_s))
                labels_tail[tags_mapping[p]].add((e_o - 1, e_s - 1))
            entity = [list(v) for v in entity]
            head = [list(v) for v in labels_head]
            tail = [list(v) for v in labels_tail]
            # labels = [list(v) for v in entity + labels_head + labels_tail]
            # for v in labels:
            #     if not v:
            #         v.append((0, 0))
            all_input_idx.append(input_idx)
            all_entity.append(entity)
            all_head.append(head)
            all_tail.append(tail)
            # all_lables.append(labels)
            # all_lables = padding_seq(all_lables, [0, 0])
            all_segments.append(segments)
            all_tokens.append(json.dumps(tokens, ensure_ascii=False))
            all_offset.append(offset)
            all_spo_offset.append(json.dumps(cur_spo_offset, ensure_ascii=False))
            all_spo.append(json.dumps(spo, ensure_ascii=False))
            all_text.append(txt)
            all_spo_ori.append(json.dumps(spo_ori, ensure_ascii=False))
            all_ori2new_idx.append(json.dumps(ori2new_idx, ensure_ascii=False))
            all_new2ori_idx.append(json.dumps(new2ori_idx, ensure_ascii=False))
            offset += stride
            # 拆分保存 
            if len(all_input_idx) > nums_save:
                features = (all_input_idx, all_segments, all_entity, all_head, all_tail, all_tokens, all_offset, all_spo_offset, all_spo, all_text, all_spo_ori, all_ori2new_idx, all_new2ori_idx)
                writejson(features, f"{path_save}_{nums_file}")
                all_input_idx = []
                # all_lables = []
                all_entity = []
                all_head = []
                all_tail = []
                all_segments = []
                all_tokens = []
                all_offset = []
                all_spo_offset = []
                all_spo = []
                all_text = []
                all_spo_ori = []
                all_ori2new_idx = []
                all_new2ori_idx = []
                nums_file += 1
    if len(all_input_idx) > 0:
        # all_lables = padding_seq(all_lables, [0, 0])
        features = (all_input_idx, all_segments, all_entity, all_head, all_tail, all_tokens, all_offset, all_spo_offset, all_spo, all_text, all_spo_ori, all_ori2new_idx, all_new2ori_idx)
        writejson(features, f"{path_save}_{nums_file}")
        nums_file += 1
    return [f'{path_save}_{i}' for i in range(nums_file)]


def collate_fn_duie_gp(datas):
    ans = [[] for _ in range(len(datas[0]))]
    for i in range(len(datas[0])):
        if i < 2:
            max_len = max([len(data[i]) for data in datas])
            ans[i].extend([data[i] + [0] * (max_len - len(data[i])) for data in datas])
        elif i < 5:
            max_len = max([max([len(v) for v in data[i]]) for data in datas])
            for data in datas:
                cur = []    
                for sub in data[i]:
                    cur.append(sub + [[0, 0]] * (max_len - len(sub)))
                ans[i].append(cur)
        else:
            ans[i] = [data[i] for data in datas]
    ans[:5] = [torch.tensor(v, dtype=torch.long) for v in ans[:5]]
    return ans

def get_data_loader(paths, bert_cfg, batch_size, max_len, data_tag, tags, file_handled, file_features, cache_dir='./cache', nums_save=10000, shuffle=True, features_num=-1, stride=64, label_type='pre', do_lower_case=False):
    dir_files = get_file_paths(cache_dir)
    is_handled, is_features = False, False
    for p in dir_files:
        if f'{file_handled}_{data_tag}' in p:
            is_handled = True
        if f'{file_features}_{data_tag}' in p:
            is_features = True
    datas_handled = None
    if not is_handled:
        datas = readdata(paths)
        datas_handled = handle_data(datas, bert_cfg, features_num, do_lower_case)
        split_save(datas_handled, f'{cache_dir}/{file_handled}_{data_tag}', nums_save)
    else:
        if not is_features:
            datas_handled = read_data_handled(f'{file_handled}_{data_tag}', cache_dir)
    features_paths = []
    if not is_features:
        tokenizer = bert_tokenize(bert_cfg) 
        features_paths = generate_features(datas_handled, max_len, len(tags), {tag: i for i, tag in enumerate(tags)}, tokenizer, features_num, nums_save, f'{cache_dir}/{file_features}_{data_tag}', stride, label_type)
    else:
        for p in dir_files:
            if f'{file_features}_{data_tag}' in p:
                features_paths.append(p)
    return DataLoader(FileDatasetPad(features_paths, shuffle), batch_size, collate_fn=collate_fn_duie_gp)
        
def collate_fn(data_batch):
    res = [[] for _ in range(len(data_batch[0]))]
    for i in range(len(data_batch[0])):
        if i < 2:
            max_len = max([len(v[i]) for v in data_batch])
            for sub in data_batch:
                # sub[i].extend([0] * (max_len - len(sub[i])))
                res[i].append(sub[i] + [0] * (max_len - len(sub[i])))
        else:
            max_len = max([max([len(d) for d in v[i]]) for v in data_batch])
            for sub in data_batch:
                cur = []
                for d in sub[i]:
                    # d.extend([0, 0] * (max_len - len(d)))
                    cur.append(d + [[0, 0]] * (max_len - len(d)))
                res[i].append(cur)
    
    return [torch.tensor(v, dtype=torch.long) for v in res]

def get_data_bert4keras(filename, bacth_size, cache_dir='./cache', nums_file=-1):
    paths = [ p for p in get_file_paths(cache_dir) if filename in p]
    if nums_file > 0:
        paths = paths[:nums_file]
    return DataLoader(FileDatasetPad(paths, True, 5), bacth_size, collate_fn=collate_fn)