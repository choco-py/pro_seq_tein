import pickle
import pandas as pd
import numpy as np
import glob
# import src.data_loader as data_preprocessor
import os
import tensorflow as tf
path = os.getcwd()
print(path)
import importlib

now = path.split('/')[-1]
if now != 'pro_seq_tein':
    os.chdir('/home/hsohee/git/pro_seq_tein')
print(os.getcwd())
import src.model as ProteinSeq

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

one_hot = pd.get_dummies(protein_data['protein_name'])
protein_data['label']=list(np.array(one_hot))
# %%

importlib.reload(ProteinSeq)
tf.reset_default_graph()
Protein_Seq = ProteinSeq.ProteinSeq
protein_1st = Protein_Seq(
    input_shape=(None, 403, 768),
    label_shape=(None, 488),
    batch_size=128,
    buffer_size=1000,
    dropout=1.0,
)
# %%
protein_1st.train(
    input_=protein_data['BERT'].tolist(),
    input_label=protein_data['label'].tolist(),
    batch_size=128,
    epoch_num=100,
    dropout=0.7,
    model_save_dir='./model_save/1st',
    pre_trained_path='./model_save/1st',
)

# %%
sub_data = protein_data.iloc[:200]
pred = protein_1st.evaluate(
    input_protein=sub_data['BERT'].tolist(),
    pre_trained_path='./model_save/1st',
)
# %%
sub_data2 = protein_data.iloc[200:]
pred2 = protein_1st.evaluate(
    input_protein=sub_data2['BERT'].tolist(),
    pre_trained_path='./model_save/1st',
)
pred.shape
all = np.append(pred, pred2)
all
tt= all.reshape(-1, 488)
protein_data['OUTPUT'] = tt.tolist()
with open('./data/pred_data.pickle', 'wb') as f:
    pickle.dump(protein_data, f, pickle.HIGHEST_PROTOCOL)


import pandas as pd

import os

os.getcwd()

os.getcwd()
os.getcwd()
os.chdir('/Users/soheehwang/git/2nd_siameseNet/data/similar_target')
tt = pd.read_csv('PRMTs_from_Integrity_DB.csv')
tt.groupby('Category').count()
