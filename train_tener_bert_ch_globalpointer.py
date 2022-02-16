# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3

from functools import cache
import logging
import os
from numpy import mod, number
from tqdm import tqdm
import torch
from torch import optim
from torch.nn.functional import dropout
# from torch.optim.lr_scheduler import

from data_preprocess.cluener import get_k_folder_dataloder
from my_py_toolkit.file.file_toolkit import readjson, make_path_legal
from my_py_toolkit.data_visulization.tensorboard import visual_data, visual_tensorboard
from my_py_toolkit.log.logger import get_logger
from models.TENER_BERT_globalpointer import TENER
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
    tags_path = os.path.join(data_dir, 'tags.json')
    bert_cfg_path = './resources/bert_model/bert'
    bert_cfg = readjson(os.path.join(bert_cfg_path, 'bert_config.json'))
    max_len = 64 # bert_cfg.get('max_position_embeddings')
    dim_embedding = bert_cfg.get('hidden_size')
    batch_size = 70
    epochs = 6
    number_layer = 1
    d_model = 768 
    heads_num = 128      
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
    
    device = 'cuda'
    dir_saved_model = './cache/model/'
    
    log_dir_tsbd = '../log/tener_log/'
    log_dir = '../log/run.log'
    data_paths = [os.path.join(data_dir, 'train.json'), os.path.join(data_dir, 'dev.json')]
    
    split_label = True
    
    tags = readjson(tags_path)
    tags_mapping = {idx:tag for idx, tag in enumerate(readjson(tags_path)) } 
    
    for p in [dir_saved_model, log_dir_tsbd, data_dir, cache_dir, log_dir]:
        make_path_legal(p)
    
    logger = get_logger(log_dir)
    
    # todo: 添加参数
    # todo: 加载模型可以写个封装方法
    model = TENER(tags_mapping, bert_cfg_path, dim_embedding, number_layer, d_model, heads_num, dim_feedforward,
                  dropout).to(device)
    
    
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
    
    # torch.optim
    losses = []
    for folder, (train_data, test_data) in enumerate(get_k_folder_dataloder(data_paths, bert_cfg_path, tags_path, max_len, split_label, batch_size, cache_dir)):
        # 全训练机器太慢，只训练部分。
        if folder > 0:
            break
        scheduler = get_linear_warmup(opt, warmup, len(train_data) * epochs)
        for epoch in range(epochs):
            model.train()
            for step, (input_idx, label_idx, segments, mask) in tqdm(enumerate(train_data)):
                input_idx, label_idx, segments = input_idx.to(device), label_idx.to(device), segments.to(device)
                opt.zero_grad()
                out = model(input_idx, label_idx, segments)
                loss = out.get('loss')
                f1 = out.get('f1')
                visual_tensorboard(log_dir_tsbd, f'train {folder} folder', {'loss': [loss.item()], 'f1': [f1.item()]}, epoch, step)
                losses.append(loss.item())
                loss.backward()
                
                if use_fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(opt), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                opt.step()
                scheduler.step()
                logger.info(f'epoch: {epoch}, {folder} Folder, Step: {step}, loss: {loss.item()}')
            torch.save(model.state_dict(), os.path.join(dir_saved_model, f'tener_weight_{folder}_{epoch}.pkl'))
            
            
            model.eval()
            label_true, label_pre = [], []
            for step, (input_idx, label_idx, segments, _) in tqdm(enumerate(test_data)): 
                input_idx, label_idx, segments = input_idx.to(device), label_idx.to(device), segments.to(device)
                out = model.predict(input_idx, segments)
                l_pre = out.get('pred')
                label_true.extend([convert_label_idx(l, tags_mapping) for l in label_idx.tolist()])
                label_pre.extend([convert_label_idx(l, tags_mapping) for l in l_pre.tolist()])
            report = classification_report(label_true, label_pre, output_dict=True)
            logger.info(f'epoch: {epoch}, {folder} Folder: ' + str(report))
            visual_tensorboard(log_dir_tsbd, f'test_{folder} folder', get_metric(report), 1, epoch)
            
            
            
                

                
            
    

if __name__ == "__main__":
    main()