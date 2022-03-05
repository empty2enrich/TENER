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
    log_path = 'tools/files/run.log'
    res_path = './tools/gp_res.xlsx'
    res = []
    keys = ['f1', 'em', 'tp', 'tpfp', 'tpfn']
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
            if cur:
                res.append(cur)
    write_excel(res, res_path)

if __name__ == "__main__":
    handle_log_gp()