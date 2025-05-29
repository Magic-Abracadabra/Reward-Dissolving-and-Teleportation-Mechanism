import os
os.chdir('D:\\毕业论文')
files = os.listdir('48')

import pandas as pd
tables = list(map(lambda x: pd.read_csv('48\\'+x).iloc[::-1].reset_index(inplace=False).drop(columns=['index'], inplace=False).set_index('时间'), files))
n = len(tables)
columns = list(tables[0].columns)
