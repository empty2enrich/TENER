# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3



# 处理准确率

from my_py_toolkit.file.file_toolkit import read_file
from my_py_toolkit.file.excel.excel_toolkit import write_excel
import json


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

if __name__ == "__main__":
    txt = read_file('../log/run.log', '\n')
    txt = [v for v in txt if 'recall' in v]
    data = []
    for t in txt:
        t = json.loads(t[t.index('Folder: ') + 8:].replace("'", '"'))
        data.append(handle_data(t))
    write_excel(data, 'cache/test.xls')
    print('finish')