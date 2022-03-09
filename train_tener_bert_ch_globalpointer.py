# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3

# from functools import cache
from cProfile import label
import enum
import json
import logging
from msilib.schema import Error
import os
from tkinter import N
from numpy import mod, number
from tqdm import tqdm
import torch
from torch import optim, zero_
from torch.nn.functional import dropout
# from torch.optim.lr_scheduler import

from data_preprocess.cluener import get_k_folder_dataloder, get_dataloader_file, get_dataloader_test
from my_py_toolkit.file.file_toolkit import readjson, make_path_legal, writejson
from my_py_toolkit.data_visulization.tensorboard import visual_data, visual_tensorboard
from my_py_toolkit.log.logger import get_logger
from my_py_toolkit.torch.transformers_pkg import bert_tokenize
from models.TENER_BERT_globalpointer import TENER
from models.utils import f1_globalpointer, predict_globalpointer, compare_res, f1_globalpointer_sparse, em_globalpointer_sparse, predict_globalpointer_compare
from seqeval.metrics import classification_report

from torch.optim.lr_scheduler import LambdaLR



def get_linear_warmup(optizer, nums_warmup_step, nums_train_step, last_epoch=-1):
    def warmup(cur_step):
        if cur_step < nums_warmup_step:
            return cur_step / max(0, nums_warmup_step)
        return max(0, nums_train_step - cur_step)/max(1, nums_train_step - nums_warmup_step)
    return LambdaLR(optizer, warmup, last_epoch)

def get_linear_warmup_2(optizer, nums_warmup_step, nums_train_step, last_epoch=-1):
    def warmup(cur_step):
        if cur_step < nums_warmup_step:
            return cur_step / max(0, nums_warmup_step)
        return 1
    return LambdaLR(optizer, warmup, last_epoch)    

def convert_label_idx(labels_idx, labels_mapping):
    return [labels_mapping[v] for v in labels_idx]

def merge_dict(*into_dict):
    # todo: 加注释
    """"""
    res = {}
    for suffix, data in into_dict:
        for k,v in data.items():
            k = k if not suffix else f'{suffix}-{k}'
            res[k] = v
    return res

def get_metric(report):
    """"""
    res = {}
    for key in report:
        for k,v in report.get(key, {}).items():
            res[f'{key}_{k}'] = [v]
    return res

def main():
    data_dir = './resources/dataset/cluener/'
    cache_dir = './cache'
    tags_path = os.path.join(data_dir, 'tags_globalpointer.json')
    bert_cfg_path = './resources/bert_model/bert'
    bert_cfg = readjson(os.path.join(bert_cfg_path, 'bert_config.json'))
    max_len = 64 # bert_cfg.get('max_position_embeddings')
    dim_embedding = bert_cfg.get('hidden_size')
    batch_size = 50
    epochs = 10
    number_layer = 1
    d_model = 768 
    heads_num = 128      
    gp_head_size = 64
    dim_feedforward = 2048
    dropout = 0.95
    warmup = 100
    lr = 3e-5
    bert_lr = 3e-5
    other_lr = 1e-3
    weight_decay_bert = 0.0
    weight_decay_other = 0.0

    use_fp16 = False
    max_grad_norm = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir_saved_model = './cache/model/'
    
    log_dir_tsbd = '../log/tener_log/'
    log_dir = '../log/run.log'
    data_paths = [os.path.join(data_dir, 'train.json'), os.path.join(data_dir, 'dev.json')]
    test_path = os.path.join(data_dir, 'test.json')
    res_compare_path = 'res_compare.json'
    
    split_label = True
    
    # debug 参数
    is_train = True
    is_continue = True
    debug = False
    steps_debug = 1
    nums_train_data = -1
    nums_test_data = -1
    size_split = -1
    sparse = True
    use_te = True # 是否使用 transformer encoder
    model_saved = os.path.join(cache_dir, 'model/most/tener_weight_0_5.pkl')
    model_continue = os.path.join(cache_dir, 'model/tener_weight_0_9.pkl')

    tags = readjson(tags_path)
    tags_mapping = {idx:tag for idx, tag in enumerate(readjson(tags_path)) } 
    
    for p in [dir_saved_model, log_dir_tsbd, data_dir, cache_dir, log_dir]:
        make_path_legal(p)
    
    logger = get_logger(log_dir)
    
    # todo: 添加参数
    # todo: 加载模型可以写个封装方法
    model = TENER(tags_mapping, bert_cfg_path, dim_embedding, number_layer, d_model, heads_num, dim_feedforward,
                  dropout, gp_head_size=gp_head_size, use_te=use_te).to(device)
    if is_continue:
        print(f'start loading saved model ......')
        model.load_state_dict(torch.load(model_continue, map_location=device))
    
    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param = [(n,p) for n,p in model.bert.named_parameters() if p.requires_grad==True]
    other_param = [(n, p) for n,p in list(model.named_parameters()) if not n.startswith('bert') and p.requires_grad==True]
    
    optizer_grouped_param = []
    for param, cur_lr, cur_decay in zip([bert_param, other_param], [bert_lr, other_lr], [weight_decay_bert, weight_decay_other]):
        optizer_grouped_param.append({
            'params': [p for n, p in param if not any([nd in n for nd in no_decay])],
            'lr': cur_lr, 'weight_decay': cur_decay}
                                     )
        optizer_grouped_param.append({
            'params': [p for n, p in param if any([nd in n for nd in no_decay ])],
            'lr': cur_lr, 'weight_decay': 0.0
        })
        
    
    opt = torch.optim.AdamW(optizer_grouped_param, lr=lr)
    
    def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval().half()
      
    if use_fp16:
        try:
            from apex import amp
            from apex.fp16_utils import network_to_half
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, opt = amp.initialize(model, opt)
        network_to_half(model)
        model.apply(fix_bn)
    
    if is_train:
        train_model(cache_dir, tags_path, bert_cfg_path, max_len, batch_size, epochs, warmup, use_fp16, max_grad_norm, device, dir_saved_model, log_dir_tsbd, data_paths, res_compare_path, split_label, debug, steps_debug, nums_train_data, nums_test_data, size_split, sparse, tags, logger, model, opt)
    else:
        test_model(cache_dir, bert_cfg_path, max_len, batch_size, device, test_path, model_saved, tags_mapping, model, debug, steps_debug)

def test_model(cache_dir, bert_cfg_path, max_len, batch_size, device, test_path, model_saved, tags_mapping, model, debug, steps_debug):
    daseset_test = get_dataloader_test(test_path, bert_cfg_path, max_len, batch_size)
    model.load_state_dict(torch.load(model_saved, map_location=device))
    model.eval()
    res = []
    for step, (idx, input_idx, segments, txt, new2ori_idx) in enumerate(daseset_test):
        if step > steps_debug:
            break
        idx, input_idx, segments = idx.to(device), input_idx.to(device), segments.to(device)
        out = model.predict(input_idx, segments)
        pre_res = predict_globalpointer_compare(out, tags_mapping)
        res.extend(generate_res_batch(idx, txt, pre_res, new2ori_idx))
    writejson(res, os.path.join(cache_dir, 'test_predict.json'))

def generate_res_batch(idx_batch, txts_bacth, pre_res_batch, new2ori_idxs_batch):
    res = []
    for idx, txt, pre_res, new2ori_idx in zip(idx_batch, txts_bacth, pre_res_batch, new2ori_idxs_batch):
        new2ori_idx = json.loads(new2ori_idx)
        label_info = {}
        for label, (start, end) in pre_res:
            if all([v > len(new2ori_idx) for v in [start, end]]):
                continue
            end = min(end - 1, len(new2ori_idx) - 1)
            start = min(start - 1, len(new2ori_idx) - 1)
            if not label in label_info:
                label_info[label] = {}
            start_txt, end_txt = new2ori_idx[start][0], new2ori_idx[end][1]
            entity = txt[start_txt:end_txt]
            if entity in label_info[label]:
                label_info[label][entity].append([start_txt, end_txt])
            else:
                label_info[label][entity] = [[start_txt, end_txt]]
            
        res.append({'id': idx.item(), 'label': label_info})
    return res



def train_model(cache_dir, tags_path, bert_cfg_path, max_len, batch_size, epochs, warmup, use_fp16, max_grad_norm, device, dir_saved_model, log_dir_tsbd, data_paths, res_compare_path, split_label, debug, steps_debug, nums_train_data, nums_test_data, size_split, sparse, tags, logger, model, opt):
    # torch.optim
    losses = []
    train_data = get_dataloader_file([data_paths[0]], bert_cfg_path, tags_path, max_len, split_label, batch_size, cache_dir, 'global_pointer', size_split, data_type='train', limit_data=nums_train_data, valid_len=4, sparse=sparse)
    test_data = get_dataloader_file([data_paths[1]], bert_cfg_path, tags_path, max_len, split_label, batch_size, cache_dir, 'global_pointer', size_split, data_type='test', limit_data=nums_test_data, valid_len=4, sparse=sparse)
    for folder in range(1):
        # 全训练机器太慢，只训练部分。
        if folder > 0:
            break
        scheduler = get_linear_warmup(opt, warmup, len(train_data) * epochs)
        for epoch in range(epochs):
            model.train()
            for step, (input_idx, segments, label_idx, mask, _, _, _, _, _) in tqdm(enumerate(train_data)):
                if debug and step> steps_debug - 1:
                    break
                input_idx, label_idx, segments = input_idx.to(device), label_idx.to(device), segments.to(device)
                opt.zero_grad()
                out = model(input_idx, label_idx, segments, sparse)
                loss = out.get('loss')
                f1 = out.get('f1')
                # debug
                if loss.isneginf() or loss.isinf() or f1.isneginf() or f1.isinf():
                    out = model(input_idx, None, segments, sparse)
                    writejson([out.tolist(), label_idx.tolist()], f'{cache_dir}/out.json')
                    raise Exception('-inf')
                    
                logger.info(f'epoch: {epoch}, {folder} Folder, Step: {step}, loss: {loss.item()}, f1: {f1.item()}')
                visual_tensorboard(log_dir_tsbd, f'train {folder} folder', {'loss': [loss.item()], 'f1': [f1.item()]}, epoch, step)
                losses.append(loss.item())
                loss.backward()
                
                if use_fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(opt), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                opt.step()
                scheduler.step()
            torch.save(model.state_dict(), os.path.join(dir_saved_model, f'tener_weight_{folder}_{epoch}.pkl'))
            
            
            model.eval()
            # 避免除 0 操作。
            tp, tpfp, tpfn = 1, 1, 1
            res_compare = []
            for step, (input_idx, segments, label_idx, _, txt, tokens, labels_detail, ori2new_idx_str, new2ori_idx_str) in tqdm(enumerate(test_data)): 
                if debug and step > steps_debug - 1:
                    break
                input_idx, label_idx, segments = input_idx.to(device), label_idx.to(device), segments.to(device)
                out = model(input_idx, None, segments, sparse)
                out = out.greater(0)
                if sparse:
                    _, tp, tpfp, tpfn = f1_globalpointer_sparse(label_idx, out, tp, tpfp, tpfn)
                else:
                    _, tp, tpfp, tpfn = f1_globalpointer(label_idx, out, tp, tpfp, tpfn)
                    
                res_compare.extend(compare_res(out, label_idx, tags, txt, [json.loads(idxstr) for idxstr in new2ori_idx_str], True))
                
            
            f1_avg = 2* tp/(tpfp + tpfn)
            em_avg = tp/tpfp
            logger.info(f'epoch: {epoch}, {folder} Folder: em: {em_avg}, f1:{f1_avg}, tp:{tp}, tpfp:{tpfp}, tpfn:{tpfn}')
            writejson(res_compare, os.path.join(cache_dir, res_compare_path + f'_{epoch}'))
            visual_tensorboard(log_dir_tsbd, f'test_{folder} folder', {'em':[em_avg.item()], 'f1':[f1_avg.item()]}, 1, epoch)
            
            
            
                

                
            
    

if __name__ == "__main__":
    main()