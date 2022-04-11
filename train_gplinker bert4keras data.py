# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3


from asyncio.log import logger
from distutils.log import debug
import json
from operator import mod
import os
from pickle import TRUE
from cv2 import log
import torch

# from data_preprocess.duie import get_data_loader, get_data_bert4keras
from data_preprocess.duie_gp_torch import get_dataload
from my_py_toolkit.data_visulization.tensorboard import visual_tensorboard
from my_py_toolkit.decorator.decorator import fn_timer
from my_py_toolkit.file.file_toolkit import readjson, writejson
from my_py_toolkit.log.logger import get_logger
from my_py_toolkit.torch.utils import get_linear_warmup
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
from tqdm import tqdm
from models.gplinker_bert4keras import GPLinker
from models.utils import f1_globalpointer_sparse

from transformers.modeling_bert import BertModel

# def get_linear_warmup(optizer, nums_warmup_step, nums_train_step, last_epoch=-1):
#     def warmup(cur_step):
#         if cur_step < nums_warmup_step:
#             return cur_step / max(0, nums_warmup_step)
#         return max(0, nums_train_step - cur_step)/max(1, nums_train_step - nums_warmup_step)
#     return LambdaLR(optizer, warmup, last_epoch)

# def get_linear_warmup_2(optizer, nums_warmup_step, nums_train_step, last_epoch=-1):
#     def warmup(cur_step):
#         if cur_step < nums_warmup_step:
#             return cur_step / max(0, nums_warmup_step)
#         return 1
#     return LambdaLR(optizer, warmup, last_epoch)    

@fn_timer()
def predict_gplinker(out, tags_id2name):
    entity, head, tail = out.split((2, len(tags_id2name), len(tags_id2name)), 1)
    idx = (entity > 0 ).nonzero().tolist()
    pre_entity = [[set() for i in range(2)] for _ in range(out.shape[0])]
    pre = [set() for _ in range(out.shape[0])]
    for batch, i, start, end in idx:
        pre_entity[batch][i].add((start, end))
    for batch, (object, subject) in enumerate(pre_entity):
        for s_o, e_o in object:
            for s_s, e_s in subject:
                idx_head = (head[batch, :, s_o, s_s] > 0).nonzero().squeeze(1).tolist()
                idx_tail = (tail[batch, :, e_o, e_s] > 0).nonzero().squeeze(1).tolist()
                # print(s_o, e_o, s_s, e_s, idx_head, idx_tail)
                for p in set(idx_head) & set(idx_tail):
                    pre[batch].add((s_o, e_o + 1, tags_id2name[p], s_s, e_s + 1))
    return pre


def trans_ids(idx, new2ori, txt):
    if idx > len(new2ori) - 1:
        ori = new2ori[-1][1]
    else:    
        ori = new2ori[idx][0]
    while ori < len(txt) and not txt[ori]:
        ori += 1
    return ori

def convert(spo, txt, offset, new2ori):
    s_o, e_o, p, s_s, e_s = spo
    s_o, e_o, s_s, e_s = [trans_ids(v + offset, new2ori, txt) for v in [s_o, e_o, s_s, e_s]]
    return (txt[s_o:e_o], p, txt[s_s:e_s])

def gp_linker_metric(out, spo_offset, tags_id2name, tp=0, tpfp=0, tpfn=0, res_compare=[]):
    
    spo_pre = predict_gplinker(out, tags_id2name)
    spo_true = [json.loads(v) for v in spo_offset]
    for y_true, y_pre in zip(spo_true, spo_pre):
        new2ori = json.loads(new2ori)
        y_true = [(s_o, e_o, p.split(',')[0], s_s, e_s) for s_o, e_o, p, s_s, e_s in y_true]
        
        tp += len(set(y_true) & set(y_pre))
        tpfp += len(y_pre)
        tpfn += len(y_true)
        # res_compare.append({'txt':txt, 'new': list(y_pre - y_true), 'lack': list(y_true - y_pre), 
        #                     'true': list(y_true), 'pre': list(y_pre)})
        
    return tp, tpfp, tpfn, res_compare

@fn_timer()
def get_label_gplinker(spo_batch):
    res = [set() for _ in range(len(spo_batch))]
    for i, item in enumerate(spo_batch):
        for s_o, e_o, p, s_s, e_s in item:
            res[i].add((s_o, e_o, p, s_s, e_s))
    return res

def main():
    ###################################### 超参数 ############################################
    cache_dir = './cache'
    data_dir = './resources/dataset/duie'
    tags_path = f'{data_dir}/tags_bert4keras.json'
    bert_cfg_path = './resources/bert_model/bert'
    dir_saved_model = f'{cache_dir}/model/'
    bert_cfg = readjson(f'{bert_cfg_path}/bert_config.json')
    log_dir_tsbd = '../log/gp_linker/'
    log_dir = '../log/run.log'
    logger = get_logger(log_dir)
    # data_paths = [os.path.join(data_dir, 'train_data.json'), os.path.join(data_dir, 'dev_data.json')]
    train_path = 'F:/Study/Github/GPLinker_pytorch/data_caches/spo/spo/1.0.0/4f608ff3259ef2cd7e69d8dd58a3aa33a4f4e1fef81d944e417267137f59e237/cache-train-bert-128-resources_bert_model_bert.arrow'
    # test_path = os.path.join(data_dir, 'test_data.json')
    test_path = 'F:/Study/Github/GPLinker_pytorch/data_caches/spo/spo/1.0.0/4f608ff3259ef2cd7e69d8dd58a3aa33a4f4e1fef81d944e417267137f59e237/cache-dev-bert-128-resources_bert_model_bert.arrow'
    file_handled = 'duie_handled.json'
    file_features = 'duie_features.json'    
    schemas_file = 'all_50_chemas'
    label_type = 'pre' # 表示只使用 pre, 否则就使用 object_type, p, subject_type
    stride = 64
    res_compare_path = f'{cache_dir}/duie_res_compare.json'
    # features 按 nums save 拆开存储
    nums_save = 3000

    batch_size = 5
    max_len = 128
    epochs = 20
    lr = 3e-5
    lr_bert = 3e-5
    lr_other = 3e-5
    weight_decay_bert = 0.0
    weight_decay_other = 0.0
    warmup = 0.1
    dim_embedding = bert_cfg['hidden_size']
    dim_model = 512
    gp_head_size = 64
    dropout = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tags = readjson(tags_path)
    tags_mapping = {i: tag for i, tag in enumerate(tags)}
    
    # debug 参数
    is_train = True
    is_continue = False
    debug = False
    steps_debug = 10
    features_num = 1000
    nums_file = 1
    # use_te = False
    model_continue = 'gplinker_weight_19.pth'
    
    
    # model
    model = GPLinker(tags_mapping, bert_cfg_path, dim_embedding, dim_model, gp_head_size).to(device)
    # 打印模型结构
    # summary(model, [(1, max_len), (1, 34, 2), (1, max_len)])
    # summary(model.entity, (batch_size, max_len, dim_embedding))
    # summary(model.head, (batch_size, max_len, dim_embedding))
    # summary(model.tail, (batch_size, max_len, dim_embedding))
    # summary(model.entity, (max_len, dim_embedding))
    if is_continue:
        model.load_state_dict(torch.load(f'{dir_saved_model}{model_continue}'))
    # 优化器
    # no_decay = ['bias', 'LayerNorm.weight']
    # bert_param = [(n,p) for n,p in model.bert.named_parameters() if p.requires_grad==True]
    # other_param = [(n, p) for n,p in list(model.named_parameters()) if not n.startswith('bert') and p.requires_grad==True]
    
    # optizer_grouped_param = []
    # for param, cur_lr, cur_decay in zip([bert_param, other_param], [lr_bert, lr_other], [weight_decay_bert, weight_decay_other]):
    #     optizer_grouped_param.append({
    #         'params': [p for n, p in param if not any([nd in n for nd in no_decay])],
    #         'lr': cur_lr, 'weight_decay': cur_decay}
    #                                  )
    #     optizer_grouped_param.append({
    #         'params': [p for n, p in param if any([nd in n for nd in no_decay ])],
    #         'lr': cur_lr, 'weight_decay': 0.0
    #     }) 
    # opt = torch.optim.AdamW(optizer_grouped_param, lr=lr)
    
    # todo: debug, 修改优化器
    all_params = [p for p in model.parameters() if p.requires_grad==True]
    opt = torch.optim.Adam(all_params, lr=lr)
    if is_train:
        # todo: 加载 dataloader.
        # data_train = get_data_bert4keras('train_', batch_size, './cache/bert4keras', nums_file)
        # data_test = get_data_bert4keras('test_', batch_size, './cache/bert4keras', nums_file)
        data_train = get_dataload(train_path, bert_cfg, len(tags_mapping), batch_size, 1, True)
        data_test = get_dataload(test_path, bert_cfg, len(tags_mapping), batch_size, 1, True)
        # data_train = get_data_loader(data_paths[:1], bert_cfg_path, batch_size, max_len, 'train', tags, file_handled, file_features, cache_dir, nums_save, True, features_num, stride, label_type)
        # data_test = get_data_loader(data_paths[1:], bert_cfg_path, batch_size, max_len, 'dev', tags, file_handled, file_features, cache_dir, nums_save, True, features_num, stride, label_type)
        steps_warmup = int(len(data_train) * epochs * warmup / batch_size)
        scheduler = get_linear_warmup(opt, steps_warmup, len(data_train) * epochs)
        total_loss = 0.0
        steps_global = 0
        
        for epoch in range(epochs):
            # todo : debug
            # process_bar = tqdm(range(len(data_train) // batch_size + 1), desc='Trainning ')
            # model.train()
            # for step, batch in enumerate(data_train):
            #     if debug and steps_debug > -1 and step > steps_debug:
            #         break
            #     batch = [v.to(device) for v in batch]
            #     input_idx, segments, entity, head, tail = batch
            #     # input_idx = input_idx.to(device)
            #     # label_idx = label_idx.to(device)
            #     # segments = segments.to(device)
            #     out = model(input_idx, segments, entity, head, tail, True)
            #     loss = out['loss']
            #     total_loss += loss.item()
            #     steps_global += 1
            #     process_bar.update(1)
            #     process_bar.set_description(f'Trainning: avg loss: {total_loss / steps_global}, loss: {loss.item()}')
            #     out_info = ', '.join([f'{k}: {v.item():.4f}' for k, v in out.items()])
            #     # logger.info(f'epoch: {epoch}, step: {step}, {out_info}')
            #     out_info = {k: [v.item()] for k,v in out.items()}
            #     visual_tensorboard(log_dir_tsbd, f'train : ', out_info, epoch, step)
            #     loss.backward()
            #     scheduler.step()
            #     opt.step()
            # torch.save(model.state_dict(), f'{dir_saved_model}/gplinker_weight_{epoch}.pth')    
            # process_bar.close()
            
            
            model.eval()
            tp, tpfp, tpfn = 1, 1, 1
            res_compare = []
            test_bar = tqdm(range(len(data_test)))
            for step, batch in enumerate(data_test):
                if debug and steps_debug > -1 and step > steps_debug:
                    break
                batch = [v.to(device) for v in batch]
                input_idx, segments, entity, head, tail = batch
                # input_idx = input_idx.to(device)
                # label_idx = label_idx.to(device)
                # segments = segments.to(device)
                e_pre, h_pre, t_pre = model.predict(input_idx, segments)
                tp, tpfp, tpfn, res_compare = gp_linker_metric([e_pre, h_pre, t_pre], [entity, head, tail], tags_mapping)
                test_bar.update(1)
                test_bar.set_description(f'tp: {tp}, tpfp: {tpfp}, tpfn: {tpfn}')
                # pre_res = predict_gplinker(out, {i:v for i, v in enumerate(tags)})
                # true_res = get_label_gplinker([json.loads(v) for v in spo])
                # for y_pre, y_true in zip(pre_res, true_res):
                #     tp += len(y_pre & y_true)
                #     tpfp += len(y_pre)
                #     tpfn += len(y_true)
                # for y_pre, y_true in zip([e_pre, h_pre, t_pre], [entity, head, tail]):
                #     _, tp, tpfp, tpfn = f1_globalpointer_sparse(y_true, y_pre, tp, tpfp, tpfn)
                # todo: res compare
                # logger.info(f'epoch: {epoch}, step: {step}, tp: {tp}, tpfp: {tpfp}, tpfn: {tpfn}')
            
            test_bar.close()    
            f1_avg = 2* tp/(tpfp + tpfn)
            em_avg = tp/tpfp
            logger.info(f'epoch: {epoch}, : em: {em_avg}, f1:{f1_avg}, tp:{tp}, tpfp:{tpfp}, tpfn:{tpfn}')
            writejson(res_compare, os.path.join(cache_dir, res_compare_path + f'_{epoch}'))
            visual_tensorboard(log_dir_tsbd, f'test', {'em':[em_avg], 'f1':[f1_avg]}, 1, epoch)
                
    else:
        pass
    

if __name__ == '__main__':
    main()

