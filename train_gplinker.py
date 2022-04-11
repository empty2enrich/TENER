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
from more_itertools import zip_equal
import torch

from data_preprocess.duie import get_data_loader
from my_py_toolkit.data_visulization.tensorboard import visual_tensorboard
from my_py_toolkit.decorator.decorator import fn_timer
from my_py_toolkit.file.file_toolkit import readjson, writejson
from my_py_toolkit.log.logger import get_logger
from my_py_toolkit.torch.utils import get_linear_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from models.gplinker import GPLinker
from models.utils import f1_globalpointer_sparse

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
    entity, head, tail = out
    batch_size = entity.shape[0]
    idx = (entity > 0 ).nonzero()
    pre_entity = [[set() for i in range(2)] for _ in range(batch_size)]
    pre = [set() for _ in range(batch_size)]
    for batch, i, start, end in idx:
        pre_entity[batch][i].add((start, end))
    for batch, (object, subject) in enumerate(pre_entity):
        for s_o, e_o in object:
            for s_s, e_s in subject:
                idx_head = (head[batch, :, s_o, s_s] > 0).nonzero().squeeze(1)
                idx_tail = (tail[batch, :, e_o, e_s] > 0).nonzero().squeeze(1)
                # print(s_o, e_o, s_s, e_s, idx_head, idx_tail)
                for p in set(idx_head) & set(idx_tail):
                    pre[batch].add((s_o.item(), e_o.item() + 1, tags_id2name[p], s_s.item(), e_s.item() + 1))
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

def gp_linker_metric(out, spo_offset, offsets, txts, new2ori_ids, tags_id2name, tp=0, tpfp=0, tpfn=0, res_compare=[]):
    
    spo_pre = predict_gplinker(out, tags_id2name)
    spo_true = [json.loads(v) for v in spo_offset]
    for y_true, y_pre, offset, txt, new2ori in zip(spo_true, spo_pre, offsets, txts, new2ori_ids):
        new2ori = json.loads(new2ori)
        y_true = [(s_o, e_o, p.split(',')[0], s_s, e_s) for s_o, e_o, p, s_s, e_s in y_true]
        y_true = set([convert(v, txt, offset, new2ori) for v in y_true])
        y_pre = set([convert(v, txt, offset, new2ori) for v in y_pre])
        
        tp += len(set(y_true) & set(y_pre))
        tpfp += len(y_pre)
        tpfn += len(y_true)
        res_compare.append({'txt':txt, 'new': list(y_pre - y_true), 'lack': list(y_true - y_pre), 
                            'true': list(y_true), 'pre': list(y_pre)})
        
    return tp, tpfp, tpfn, res_compare    

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
    data_paths = [os.path.join(data_dir, 'train_data.json'), os.path.join(data_dir, 'dev_data.json')]
    test_path = os.path.join(data_dir, 'test_data.json')
    file_handled = 'duie_handled.json'
    file_features = 'duie_features.json'    
    do_lower_case = True
    schemas_file = 'all_50_chemas'
    label_type = 'pre' # 表示只使用 pre, 否则就使用 object_type, p, subject_type
    stride = 64
    res_compare_path = f'{cache_dir}/duie_res_compare.json'
    # features 按 nums save 拆开存储
    nums_save = 3000

    batch_size = 5
    batch_size_test = 1
    max_len = 128
    epochs = 20
    lr = 2e-5
    lr_bert = 3e-5
    lr_other = 3e-5
    weight_decay_bert = 0.0
    weight_decay_other = 0.0
    warmup = 500
    dim_embedding = bert_cfg['hidden_size']
    dim_model = 512
    gp_head_size = 64
    dropout = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tags = readjson(tags_path)
    tags_mapping = {i: tag for i, tag in enumerate(tags)}
    steps_eval = 20
    
    # debug 参数
    is_train = True
    is_continue = True
    debug = False
    steps_debug = 1
    features_num = 1000
    # use_te = False
    model_continue = 'gplinker_weight_19.pth'
    
    writer = SummaryWriter(log_dir_tsbd)
    # model
    model = GPLinker(tags_mapping, bert_cfg_path, dim_embedding, dim_model, gp_head_size).to(device)
    # 打印模型结构
    # summary(model, [(1, max_len), (1, 34, 2), (1, max_len)])
    # summary(model.entity, (batch_size, max_len, dim_embedding))
    # summary(model.head, (batch_size, max_len, dim_embedding))
    # summary(model.tail, (batch_size, max_len, dim_embedding))
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
        data_train = get_data_loader(data_paths[:1], bert_cfg_path, batch_size, max_len, 'train', tags, file_handled, file_features, cache_dir, nums_save, True, features_num, stride, label_type, do_lower_case)
        data_test = get_data_loader(data_paths[1:], bert_cfg_path, batch_size_test, max_len, 'dev', tags, file_handled, file_features, cache_dir, nums_save, True, features_num, stride, label_type, do_lower_case)
        scheduler = get_linear_warmup(opt, warmup, len(data_train) * epochs)
        
        loss_total, loss_pre, steps_avg, steps_global = 0, 0, 100, 0
        f1_max = 0
        num_eval = 0
        for epoch in range(epochs):
            
            model.train()
            p_bar = tqdm(range(len(data_train)), 'Train:')
            for step, (input_idx, segments, entity, head, tail, *_) in enumerate(data_train):
                steps_global += 1   
                if debug and steps_debug > -1 and step > steps_debug:
                    break
                opt.zero_grad()
                input_idx = input_idx.to(device)
                segments = segments.to(device)
                entity = entity.to(device)
                head =  head.to(device)
                tail = tail.to(device)
                out = model(input_idx, segments, entity, head, tail, True)
                loss = out['loss']
                loss_total += loss.item()
                steps_peroid = steps_global % steps_avg if steps_global % steps_avg > 0 else steps_avg
                writer.add_scalars(F'train/loss_{epoch}',{'loss': loss.item(), 'avg_loss': loss_total / steps_global, 
                                                 'loss_peroid': (loss_total - loss_pre)/steps_peroid}, step)
                if steps_global % steps_avg == 0:
                    loss_pre = loss_total
                p_bar.update(1)
                p_bar.set_description(f'epoch: {epoch}, step: {step}, loss: {loss.item():.4f}, avg_loss: {loss_total / steps_global:.4f}, loss_peroid: {(loss_total - loss_pre)/steps_peroid:.4f}')
                # out_info = ', '.join([f'{k}: {v.item():.4f}' for k, v in out.items()])
                # logger.info(f'epoch: {epoch}, step: {step}, {out_info}')
                # out_info = {k: [v.item()] for k,v in out.items()}
                # visual_tensorboard(log_dir_tsbd, f'train : ', out_info, epoch, step)
                
                loss.backward()
                opt.step()
                scheduler.step()
                if (steps_global + 1) % steps_eval == 0:
                    # todo: debug
                    f1 = evaluate(writer, logger, device, tags, debug, steps_debug, model, data_train, epoch, step, num_eval)
                    num_eval += 1
                    # todo: 比较 f1, 若效果更好则保存模型
                    if f1 > f1_max:
                        torch.save(model.state_dict(), f'{dir_saved_model}/gplinker_weight_{epoch}_{step}_{f1:.4f}.pth')
                        f1_max = f1    
            p_bar.close()
            
            
            # tp.append(sum(tp))
            # tpfp.append(sum(tpfp))
            # tpfn.append(sum(tpfn))
            # f1_avg = [ 2 * tp[i] /(tpfp[i] + tpfn[i]) for i in range(4)]  
            # em_avg = [tp[i]/tpfp[i] for i in range(4)]
            # metric_info = ', '.join([f'{name}_f1: {f1:.4f}, {name}_em: {em:.4f}, {name}_tp: {tp_cur}, {name}_tpfp: {tpfp_cur}, {name}_tpfn: {tpfn_cur}' for name, f1, em, tp_cur, tpfp_cur, tpfn_cur in zip(['entity', 'head', 'tail', 'all'], f1_avg, em_avg, tp, tpfp, tpfn)])
            # logger.info(f'epoch: {epoch}, {metric_info}')
            # writejson(res_compare, os.path.join(cache_dir, res_compare_path + f'_{epoch}'))
            # visual_info = { }
            # for name, em, f1, tp_cur, tpfp_cur, tpfn_cur in zip(['entity', 'head', 'tail', 'all'], em_avg, f1_avg, tp, tpfp, tpfn):
            #     visual_info[f'{name}_em'] = [em.item()]
            #     visual_info[f'{name}_f1'] = [f1.item()]
            #     visual_info[f'{name}_tp'] = [tp_cur.item()]
            #     visual_info[f'{name}_tpfp'] = [tpfp_cur.item()]
            #     visual_info[f'{name}_tpfn'] = [tpfn_cur.item()]
            # visual_tensorboard(log_dir_tsbd, f'test', visual_info, 1, epoch)
                
    else:
        pass

def evaluate(writer, logger, device, tags, debug, steps_debug, model, data_test, epoch, step, num_eval):
    model.eval()
    tp, tpfp, tpfn = 0, 0, 0
    res_compare = []
    p_par = tqdm()
    for step, (input_idx, segments, entity, head, tail, _, offsets, spo_offset, _, txt, _, _, new2ori_idx) in enumerate(data_test):
        if debug and steps_debug > -1 and step > steps_debug:
            break
        input_idx = input_idx.to(device)
        segments = segments.to(device)
        entity = entity.to(device)
        head = head.to(device)
        tail = tail.to(device)
        out = model(input_idx, segments)
                # pre_res = predict_gplinker(out, {i:v for i, v in enumerate(tags)})
                # true_res = get_label_gplinker([json.loads(v) for v in spo])
                # for y_pre, y_true in zip(pre_res, true_res):
                #     tp += len(y_pre & y_true)
                #     tpfp += len(y_pre)
                #     tpfn += len(y_true)
        tp, tpfp, tpfn, res_compare = gp_linker_metric(out, spo_offset, offsets, txt, new2ori_idx, {i:tag for i,tag in enumerate(tags)}, tp, tpfp, tpfn, res_compare)
        p_par.update()
        p_par.set_description(f'epoch: {epoch}, step: {step}, tp: {tp}, tpfp: {tpfp}, tpfn: {tpfn}')
                # for i, (y_true, y_pre) in enumerate(zip([entity, head, tail], out)):
                #     # todo: 不同数据计算不同 f1, 并记录
                #     _, tp[i], tpfp[i], tpfn[i] = f1_globalpointer_sparse(y_true, y_pre, tp[i], tpfp[i], tpfn[i])
                # todo: res compare
                # logger.info(f'epoch: {epoch}, step: {step}, tp: {tp}, tpfp: {tpfp}, tpfn: {tpfn}')
    p_par.close()
    f1  = 2 * tp / (tpfp + tpfn) if (tpfp + tpfn) > 0 else 0
    em = tp / tpfp if tpfp > 0 else 0
    recall = tp / tpfn if tpfn > 0 else 0
    logger.info(f'epoch: {epoch}, step: {step}, f1: {f1:.4f}, em: {em:.4f}, recall: {recall:.4f}, tp: {tp}, tpfp: {tpfp}, tpfn: {tpfn}')
    writer.add_scalars('test', {'f1': f1, 'em': em, 'recall': recall}, num_eval)
    writer.add_scalars('test', {'tp': tp, 'tpfp': tpfp, 'tpfn': tpfn}, num_eval)
    return f1

if __name__ == '__main__':
    main()

