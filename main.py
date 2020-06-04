import pickle
import pandas as pd
import numpy as np
import glob
import src.data_loader as data_preprocessor
import os
import tensorflow as tf

path = os.getcwd()
print(path)

now = path.split('/')[-1]
if now != 'pro_seq_tein':
    os.chdir('/Users/soheehwang/git/pro_seq_tein')
print(os.getcwd())

# %% Data Loader

data_check = glob.glob('./data/*.pickle')

# print('Reload dataset.')
# with open(data_check[1], 'rb') as f:
#     train_data = pickle.load(f)
#
# with open(data_check[0], 'rb') as f:
#     test_data = pickle.load(f)
#
# print('[shape]: train_data: '+ str(train_data.shape))
# print('[shape]: test_data: '+ str(test_data.shape))
# protein_data = train_data[['protein_name', 'seq', 'BERT']]
# protein_data.drop_duplicates('protein_name', inplace=True)
# with open('./data/protein_data.pickle', 'wb') as f:
#     pickle.dump(protein_data, f, pickle.HIGHEST_PROTOCOL)


with open('./data/protein_data.pickle', 'rb') as f:
    protein_data = pickle.load(f)
print('[shape]: protein_data: '+ str(protein_data.shape))

protein_data
protein_data.iloc[0].BERT.shape
