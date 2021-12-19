# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3


import json
import os
import re

import torch
from torch.utils.data.dataset import Dataset
from models.TENER_BERT import TENER
from my_py_toolkit.file.file_toolkit import read_file, readjson, make_path_legal, writejson
from my_py_toolkit.torch.transformers_pkg import bert_tokenize
from my_py_toolkit.ml.data.text_preprocess import tokenize_chinese
from my_py_toolkit.ml.ner.ner_toolkit import handle_predict


from torch.utils.data import DataLoader, TensorDataset

from train_tener_bert_ch import convert_label_idx

def handle_test_data(test_path, bert_cfg, max_len):
    input_idx = []
    segments = []
    idx = []
    txt = []
    idx_tranf = []
    tokenizer = bert_tokenize(bert_cfg)
    for line in read_file(test_path, '\n'):
        if not line:
            continue
        line = json.loads(line)
        idx.append(line['id'])
        tokens, transf = tokenize_chinese(tokenizer, line['text'], True)
        # tokens = tokenizer.tokenize(line['text'])
        tokens = tokens[:max_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        tokens += ['[PAD]'] * (max_len - len(tokens))
        segments.append([0] * max_len)
        input_idx.append(tokenizer.convert_tokens_to_ids(tokens))
        txt.append(line['text'])
        idx_tranf.append(transf)
    return idx, input_idx, segments, txt, idx_tranf
        
def get_dataloader(test_path, bert_cfg, max_len, batch_size):
    idx, input_idx, segments, txt, idx_tranf = handle_test_data(test_path, bert_cfg, max_len)
    idx, input_idx, segments = [torch.tensor(item, dtype=torch.long) for item in [idx, input_idx, segments]]
    return DataLoader(TensorDataset(idx, input_idx, segments), batch_size=batch_size), txt, idx_tranf

def generate_res(id, label_pre, tokens, txt, idx_tranf):
    # tokens = [re.sub('^##', '', t) for t in tokens]
    
    # 删除 '[CLS]' 和 '[SEP]'
    # tokens = tokens[1:-1]
    label_info = handle_predict(txt, label_pre, idx_tranf)
    # start = 0
    # cur_tag = ''
    # is_tag = False
    # for i, label in enumerate(label_pre):
    #     if label.startswith('S-'):
    #         cur_tag = label[2:].lower()
    #         cur_tag_info = label_info.get(cur_tag, {})
    #         start_txt, end_txt = idx_tranf.to_ori_scope(i-1, i)
    #         ner_cont = txt[start_txt:end_txt]
    #         cur_cont_info = cur_tag_info.get(ner_cont, [])
            
    #         cur_cont_info.append([start_txt, end_txt - 1])
    #         cur_tag_info[ner_cont] = cur_cont_info
    #         label_info[cur_tag] = cur_tag_info
    #     elif label.startswith('B-'):
    #         is_tag = True
    #         start = i
    #         cur_tag = label[2:].lower()
    #     elif label.startswith('I-'):
    #         if not is_tag or label[2:].lower() != cur_tag:
    #             is_tag = False
    #     elif label.startswith('E-'):
    #         if not is_tag or label[2:].lower() != cur_tag:
    #             is_tag = False
    #             continue
    #         cur_tag_info = label_info.get(cur_tag, {})
    #         start_txt, end_txt = idx_tranf.to_ori_scope(start-1, i)
    #         ner_cont = txt[start_txt:end_txt]
    #         cur_cont_info = cur_tag_info.get(ner_cont, [])
    #         cur_cont_info.append([start_txt, end_txt - 1])
    #         cur_tag_info[ner_cont] = cur_cont_info
    #         label_info[cur_tag] = cur_tag_info
    #     elif label.startswith('O'):
    #         is_tag = False
            
    return {'id': id, 'label': label_info}
            

def main():
    test_path = 'resources/dataset/cluener/test.json'
    model_saved = './cache/model/tener_weight_0_4.pkl'
    
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
    heads_num = 8
    dim_feedforward = 2048
    dropout = 0.90
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
    
    # logger = get_logger(log_dir)
    
    # todo: 添加参数
    # todo: 加载模型可以写个封装方法
    model = TENER(tags_mapping, bert_cfg_path, dim_embedding, number_layer, d_model, heads_num, dim_feedforward, dropout).to(device)
    tokenizer = bert_tokenize(bert_cfg_path)
    model.load_state_dict(torch.load(model_saved))
    model.to(device)
    ids, pre, tokens = [], [], []
    model.eval()
    dataloader, txt, idx_tranf = get_dataloader(test_path, bert_cfg_path, max_len, batch_size)
    for idx, input_idx, segments in dataloader:
        idx, input_idx, segments = idx.to(device), input_idx.to(device), segments.to(device)
        out = model.predict(input_idx, segments)
        l_pre = out.get('pred')
        pre.extend([convert_label_idx(l, tags_mapping) for l in l_pre.tolist()])
        tokens.extend([tokenizer.convert_ids_to_tokens(input) for input in input_idx.tolist()])
        ids.extend(idx.tolist())
    res = [generate_res(i, pre, t, txt[i], idx_tranf[i]) for i, pre, t in zip(ids, pre, tokens)]
    writejson(pre, './cache/test_pre.json')
    writejson(tokens, './cache/test_tokens.json')
    res = sorted(res, key=lambda x: x['id'])
    writejson(res, './cache/predict.json')
    
    
        
        
    
    
    

if __name__ == "__main__":
    main()
