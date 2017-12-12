# -*- coding：utf-8 -*-

import csv
import io
from hanziconv import HanziConv
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码 

num = 0
with open('tang(simplified).txt', 'w', encoding='utf-8') as f1:
    with open('tang(tradition).csv', 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            line = row[1]
            line = HanziConv.toSimplified(line)
            f1.write(line+'\n')
            