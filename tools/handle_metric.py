# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3



# 处理准确率

from my_py_toolkit.file.file_toolkit import read_file
from my_py_toolkit.file.excel.excel_toolkit import write_excel
import json
import re


def handle_data(data):
    res = []
    items = ['micro avg', 'macro avg', 'ADDRESS', 'BOOK', 'COMPANY', 'GAME', 'GOVERNMENT', 'MOVIE', 'NAME', 'ORGANIZATION', 'POSITION', 'SCENE']
    sub_key = ['f1-score', 'precision', 'recall', 'support']
    for key in items:
        cur = data[key]
        for k in sub_key:
            
            res.append(cur[k])
        if key == 'macro avg':
            res.append('')
    return res

def handle_log():
    txt = read_file('../log/run.log', '\n')
    txt = [v for v in txt if 'recall' in v]
    data = []
    for t in txt:
        t = json.loads(t[t.index('Folder: ') + 8:].replace("'", '"'))
        data.append(handle_data(t))
    write_excel(data, 'cache/metric.xls')
    print('finish')

def handle_log_gp():
    reg = 'em: (?P<em>[\\d\\.]+).*f1.*?(?P<f1>[\\d\\.]+).*?tp.*?(?P<tp>[\\d\\.]+).*?tpfp.*?(?P<tpfp>[\\d\\.]+).*?tpfn.*?(?P<tpfn>[\\d\\.]+)'
    reg_loss = 'epoch.*?(?P<epoch>\\d+).*?loss.*?(?P<loss>[\\d\\.]+)'
    log_path = 'tools/files/run.log'
    res_path = './tools/gp_res.xlsx'
    res = []
    keys = ['f1', 'em', 'tp', 'tpfp', 'tpfn']
    loss_info = [['min_loss', 'max_loss', 'avg_loss']]

    res.append(keys)
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cur = []
            if line:
                match = re.search(reg, line)
                if match:
                    print(line)
                    for k in keys:
                        cur.append(match[k])
                # loss 信息
                match = re.search(reg_loss, line)
                if match:
                    epoch = int(match['epoch'])
                    loss = float(match['loss'])
                    if epoch >= len(loss_info) - 1:
                        loss_info.append([loss, loss, loss, 1])
                    else:
                        cur_loss = loss_info[epoch + 1]
                        cur_loss[0] = min(cur_loss[0], loss)
                        cur_loss[1] = max(cur_loss[1], loss)
                        cur_loss[2] += loss
                        cur_loss[3] += 1

            if cur:
                res.append(cur)
    res[0] = loss_info[0] + res[0]
    for i, item in enumerate(res[1:]):
        cur_loss = loss_info[i + 1]
        res[i + 1] = cur_loss[:2] + [cur_loss[2]/cur_loss[3]] + res[i + 1]
    write_excel(res, res_path)

if __name__ == "__main__":
    handle_log_gp()