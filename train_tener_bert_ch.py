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
from models.TENER_BERT import TENER
from seqeval.metrics import classification_report

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
    max_len = bert_cfg.get('max_position_embeddings')
    dim_embedding = bert_cfg.get('hidden_size')
    batch_size = 4
    epochs = 10
    number_layer = 5
    d_model = 512
    heads_num = 8
    dim_feedforward = 2048
    dropout = 0.90
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
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad==True])
    # torch.optim
    losses = []
    for folder, (train_data, test_data) in enumerate(get_k_folder_dataloder(data_paths, bert_cfg_path, tags_path, max_len, split_label, batch_size, cache_dir)):
        # 全训练机器太慢，只训练部分。
        if folder > 0:
            break
        for epoch in range(epochs):
            model.train()
            for step, (input_idx, label_idx, segments, mask) in tqdm(enumerate(train_data)):
                input_idx, label_idx, segments = input_idx.to(device), label_idx.to(device), segments.to(device)
                opt.zero_grad()
                out = model(input_idx, label_idx, segments)
                loss = out.get('loss').mean()
                visual_tensorboard(log_dir_tsbd, f'train {folder} folder', {'loss': [loss.item()]}, epoch, step)
                losses.append(loss.item())
                loss.backward()
                opt.step()
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