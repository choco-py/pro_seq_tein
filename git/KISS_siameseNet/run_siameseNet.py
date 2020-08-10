import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import os
import sys
print(tf.__version__)
print(os.getcwd())
# os.chdir(
#  '/home/hsohee/git/2nd_siameseNet'
# )
os.chdir(
   '/home/hsohee/git/KISS_siameseNet'
 )
# os.chdir(
#    '/mnt/d/sohee/KISS_siameseNet'
#  )


# import unipy as up
from random import shuffle
# from rdkit.Chem import MolFromSmiles
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
# from rdkit.Chem.rdmolops import PatternFingerprint
# from sklearn.model_selection import train_test_split
# import random
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from scipy.spatial import distance
# from Bio import SeqIO
from matplotlib import colors as mcolors
# import seaborn as sns
# from collections import defaultdict
#
# from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
# from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
# from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
# from rdkit.Chem import RDKFingerprint
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import importlib
from src import data_loader as preprocessor

import pickle
# %% Preprocess Data ============================================

# importlib.reload(preprocessor)
file_list = glob.glob('./data/kinaseSARfari/*.xlsx')
# kinase_data = pd.read_excel(file_list[0])
# active_kinase = kinase_data[kinase_data.classificartion=='Active']
# %%
data_save_dir = './model_save/data/'
data_check = glob.glob(data_save_dir+'*.pickle')

# if len(data_check) == 0:
#     print('Generate dataset.')
#     train_data, test_data = preprocessor.data_preprocessor(
#         dataset=active_kinase,
#         smiles_cols='smiles',
#         finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
#         protein_cols='seq',
#         protein_type='RATIO',
#         train_ratio=0.2,
#     )
#     with open('model_save/data/train_data.pickle', 'wb') as f:
#         pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
#
#         print('Train Data Save.')
#
#     # save
#     with open('model_save/data/test_data.pickle', 'wb') as f:
#         pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
#         print('Test Data Save.')
#
# else:
print('Reload dataset.')
with open(data_check[1], 'rb') as f:
    train_data = pickle.load(f)

with open(data_check[0], 'rb') as f:
    test_data = pickle.load(f)

print('[shape]: train_data: '+ str(train_data.shape))
print('[shape]: test_data: '+ str(test_data.shape))

# train_data.BERT.iloc[0].shape

# %% Protein Attach ===========================================
# importlib.reload(preprocessor)
#
# tt = preprocessor.protein_loader(
#     input_data=test_data,
#     protein_cols='protein_name',
#     seq_cols='seq',
#     protein_path=None,
#     protein_type='RATIO_reverse',
# )
# tt['RATIO_SUM'] = tt['RATIO']+tt['RATIO_reverse']
# tt
# len(tt['RATIO_SUM'].iloc[0])
# test_data =tt
# %% protein_preprocess
#
# data_save_dir = './model_save/data/'
#
# data_check = glob.glob(data_save_dir+'*.pickle')
# with open(data_check[1], 'rb') as f:
#     ttt = pickle.load(f)
#
# with open(data_check[0], 'rb') as f:
#     fff = pickle.load(f)
#
# total = pd.concat(ttt,fff)
#
# pro = ttt[['protein_name', 'RATIO']]
# pro = pro.drop_duplicates('protein_name')
# pro2 = fff[['protein_name', 'RATIO']]
#
# total = pd.concat([pro, pro2])
# pro = total.drop_duplicates('protein_name')
#
# train_data['RATIO_20'] = train_data['RATIO']
#
# train_data.drop(columns=['RATIO'], inplace=True)
# train_data = train_data.join(pro.set_index('protein_name'), on='protein_name')
# train_data[train_data.RATIO.isna()==True]
# train_data
#
# test_data['RATIO_20'] = test_data['RATIO']
# test_data.drop(columns=['RATIO'], inplace=True)
# test_data = test_data.join(pro.set_index('protein_name'), on='protein_name')
# test_data[test_data.RATIO.isna()==True]
# %% Load blind kinase data ==================================================
file_list = glob.glob('./data/blind_kinase/*.xlsx')
blind_data = pd.read_excel(file_list[1])
blind_data2 = pd.read_excel(file_list[0])

blind_data2 = preprocessor.fingerprint_converter(
    input_data=blind_data2,
    finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
    smiles_cols='SMILES',
    finger_size=1024,
)
blind_data = preprocessor.fingerprint_converter(
    input_data=blind_data,
    finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
    smiles_cols='Smiles (Rdkit)',
    finger_size=1024,
)
# blind_data
# %% Load SKL data ===========================================================

# mmp_data = preprocessor.data_loader(
#     data_path='./data/similar_target/MMP')
# hdac_data = preprocessor.data_loader(
#     data_path='./data/similar_target/HDAC')
#
# mmp_data.reset_index(inplace=True)
# # %%
# mmp_data = preprocessor.fingerprint_converter(
#     input_data=hdac_data,
#     finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
#     smiles_cols='Canonical_Smiles',
#     finger_size=1024,
# )
# hdac_data = preprocessor.fingerprint_converter(
#     input_data=hdac_data,
#     finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
#     smiles_cols='Canonical_Smiles',
#     finger_size=1024,
# )
#
# fasta_list
# t = SeqIO.read(fasta_list[0], 'fasta').seq
#
# AAC.(t)
#
# # %%
# importlib.reload(preprocessor)
# protein_seq = preprocessor.protein_loader(
#     input_data=hdac_data,
#     protein_cols=None,
#     protein_path='./data/real_protein_sequence/',
#     protein_type='RATIO',
# )
#
# # %%
# hdac_data.join(protein_seq.set_index('Category'), on='Category')
# # %%
# skl_data = pd.concat([mmp_data, hdac_data])

# %% ====================== Sohee Version ======================================
# %% siamesemNet: Build -------------------------------------------------------
os.getcwd()

glob.glob('model_save/kinase_3rd/last_weights/*')

import importlib
# from workspaces.src.anogan import anogan_sohee_origin as ANOGAN
# os.chdir(
#    '/home/hsohee/git/2nd_siameseNet'
#  )
# %%
from src import siamese_network as SiameseNet
importlib.reload(SiameseNet)
SiameseNet = SiameseNet.SiameseNet

tf.reset_default_graph()
siamese_net = SiameseNet(

    input_finger1_shape=(None, 1024),
    input_finger2_shape=(None, 1024),
    input_finger3_shape=(None, 1024),
    input_finger4_shape=(None, 1024),
    input_finger5_shape=(None, 1024),
    input_finger6_shape=(None, 1024),
    # input_finger7_shape=(None, 166),
    # input_protein_shape=(None, 16840),
    input_protein_shape=(None, 1, 403, 768),
    buffer_size=10000000,
)
train_ok=True
# %% train


sub_data = train_data.iloc[10000:17000]
if train_ok:
    siamese_net.train_SiamNet(
        input_finger1=sub_data['PAIR'].tolist(),
        input_finger2=sub_data['FCFP'].tolist(),
        input_finger3=sub_data['ECFP'].tolist(),
        input_finger4=sub_data['RDKIT'].tolist(),
        input_finger5=sub_data['TORS'].tolist(),
        input_finger6=sub_data['PAIR'].tolist(),
        input_finger6=sub_data['PTFP'].tolist(),
        input_protein=sub_data['BERT'].tolist(),
        input_label=sub_data['dom_id'].tolist(),
        batch_size=1238,
        epoch_num=10000,
        dropout=0.7,
        model_save_dir='./model_save/kinase_6th',
        pre_trained_path='./model_save/kinase_6th',#'./model_save/reuse',
    )
# %%
test_data.drop(index=test_data[test_data.BERT.isna()].index, inplace=True)

subset_data = test_data[:10000]

test_data.columns
pred = siamese_net.evaluate_SiamNet(
    # input_finger1=subset_data['PTFP'].tolist(),
    input_finger1=subset_data['PAIR'].tolist(),
    input_finger2=subset_data['FCFP'].tolist(),
    input_finger3=subset_data['ECFP'].tolist(),
    input_finger4=subset_data['RDKIT'].tolist(),
    input_finger5=subset_data['TORS'].tolist(),
    input_finger6=subset_data['PTFP'].tolist(),
    # input_finger6=subset_data['PAIR'].tolist(),
    # input_finger7=subset_data['MACCS'].tolist(),
    input_protein=subset_data['BERT'].tolist(),
    input_label=subset_data['dom_id'].tolist(),
    dropout=1.0,
    pre_trained_path='./model_save/kinase_6th',
    target_epoch=None,
)
print(finish)
# %%
pred_list=[]
test_data.shape
for _ in range(14103):

    a = test_data.iloc[_:_+1]
    test_data.columns
    pred = siamese_net.evaluate_SiamNet(
        # input_finger1=subset_data['PTFP'].tolist(),
        input_finger1=a['PAIR'].tolist(),
        input_finger2=a['FCFP'].tolist(),
        input_finger3=a['ECFP'].tolist(),
        input_finger4=a['RDKIT'].tolist(),
        input_finger5=a['TORS'].tolist(),
        input_finger6=a['PTFP'].tolist(),
        # input_finger6=subset_data['PAIR'].tolist(),
        # input_finger7=subset_data['MACCS'].tolist(),
        input_protein=a['BERT'].tolist(),#.tolist(),
        input_label=a['dom_id'].tolist(),
        dropout=1.0,
        pre_trained_path='./model_save/kinase_6th',
        target_epoch=None,
    )
    pred_list.append(pred)
    print(_)
    pred.shape

    # 1328 까지 했음

# %%
# pred_2nd = []
test_data.shape

r=13000
for tt in range(100):
    q=r+tt
    a = test_data.iloc[q: q+1]
    test_data.columns
    pred = siamese_net.evaluate_SiamNet(
        # input_finger1=subset_data['PTFP'].tolist(),
        input_finger1=a['PAIR'].tolist(),
        input_finger2=a['FCFP'].tolist(),
        input_finger3=a['ECFP'].tolist(),
        input_finger4=a['RDKIT'].tolist(),
        input_finger5=a['TORS'].tolist(),
        input_finger6=a['PTFP'].tolist(),
        # input_finger6=subset_data['PAIR'].tolist(),
        # input_finger7=subset_data['MACCS'].tolist(),
        input_protein=a['BERT'].tolist(),#.tolist(),
        input_label=a['dom_id'].tolist(),
        dropout=1.0,
        pre_trained_path='./model_save/kinase_6th',
        target_epoch=None,
    )
    print(q)
# %%
tt = np.arange(131, 1400)

test_data.shape
for _ in tt:
    a = test_data.iloc[_*100:(_+1)*100]
    test_data.columns
    pred = siamese_net.evaluate_SiamNet(
        # input_finger1=subset_data['PTFP'].tolist(),
        input_finger1=a['PAIR'].tolist(),
        input_finger2=a['FCFP'].tolist(),
        input_finger3=a['ECFP'].tolist(),
        input_finger4=a['RDKIT'].tolist(),
        input_finger5=a['TORS'].tolist(),
        input_finger6=a['PTFP'].tolist(),
        # input_finger6=subset_data['PAIR'].tolist(),
        # input_finger7=subset_data['MACCS'].tolist(),
        input_protein=a['BERT'].tolist(),#.tolist(),
        input_label=a['dom_id'].tolist(),
        dropout=1.0,
        pre_trained_path='./model_save/kinase_6th',
        target_epoch=None,
    )
    pred_2nd.append(pred)
    print(_)
len(pred_2nd)
len(pred_2nd)

#109 / 75

1200 14000
# %%
test_df = pd.DataFrame(pred)
test_df['dom_id'] = subset_data['dom_id'].tolist()
test_df['smiles'] = subset_data['smiles'].tolist()
pred.shape
test_df[test_df.dom_id==2422]

# test_df.to_csv('./model_save/data/'+'test_data_bert.csv')
test_df.groupby('smiles').count().sort_values(0)
aa = 'CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl'
test_df[test_df.smiles==aa]

test_df[test_df.smiles==aa]
# %%

subset_data = blind_data
skl_len = blind_data.shape[0]

pred_blind = siamese_net.evaluate_SiamNet(
    input_finger1=subset_data['PTFP'].tolist(),
    input_finger2=subset_data['FCFP'].tolist(),
    input_finger3=subset_data['ECFP'].tolist(),
    input_finger4=subset_data['RDKIT'].tolist(),
    input_finger5=subset_data['TORS'].tolist(),
    input_finger6=subset_data['PAIR'].tolist(),
    input_finger7=subset_data['MACCS'].tolist(),
    input_protein=test_data['RATIO'].iloc[:skl_len].tolist(),
    input_label=subset_data.index.tolist(),
    dropout=1.0,
    pre_trained_path='./model_save/kinase_4th',
    target_epoch=None,
)
pred_blind.shape
blind_df = pd.DataFrame(pred_blind)
blind_df['dom_id'] = subset_data['Target priority'].tolist()
blind_df['smiles'] = subset_data['Smiles (Rdkit)'].tolist()
blind_df.describe()

blind_df.to_csv('./data/blind_kinase/'+'JAK_blind_8420.csv')
subset_data['PTFP'].shape

# %%
subset_data = blind_data2
skl_len = blind_data2.shape[0]

pred_blind = siamese_net.evaluate_SiamNet(
    input_finger1=subset_data['PTFP'].tolist(),
    input_finger2=subset_data['FCFP'].tolist(),
    input_finger3=subset_data['ECFP'].tolist(),
    input_finger4=subset_data['RDKIT'].tolist(),
    input_finger5=subset_data['TORS'].tolist(),
    input_finger6=subset_data['PAIR'].tolist(),
    input_finger7=subset_data['MACCS'].tolist(),
    input_protein=test_data['RATIO'].iloc[:skl_len].tolist(),
    input_label=subset_data.index.tolist(),
    dropout=1.0,
    pre_trained_path='./model_save/kinase_4th',
    target_epoch=None,
)
pred_blind.shape
blind_df = pd.DataFrame(pred_blind)
blind_df['dom_id'] = subset_data['Selectivity'].tolist()
blind_df['smiles'] = subset_data['SMILES'].tolist()
blind_df.describe()

blind_df.to_csv('./data/blind_kinase/'+'ROCK_blind_8420.csv')
subset_data['PTFP'].shape
# %%

subset_data = skl_data
skl_len = skl_data.shape[0]

pred_skl = siamese_net.evaluate_SiamNet(
    input_finger1=list(subset_data['PTFP']),
    input_finger2=list(subset_data['FCFP']),
    input_finger3=list(subset_data['ECFP']),
    input_finger4=list(subset_data['RDKIT']),
    input_finger5=list(subset_data['TORS']),
    input_finger6=list(subset_data['PAIR']),
    input_finger7=list(subset_data['MACCS']),
    input_protein=list(test_data['RATIO'].iloc[:skl_len]),
    input_label=list(subset_data['Name']),
    dropout=1.0,
    pre_trained_path='./model_save/kinase_2nd',
    target_epoch=None,
)
pred_skl.shape
skl_df = pd.DataFrame(pred_skl)
skl_df['dom_id'] = list(subset_data['Category'])
skl_df['smiles'] = list(subset_data['Canonical_Smiles'])
skl_df.describe()

skl_df.to_csv('./model_save/data/'+'skl_data_2nd.csv')

# %%
subset_data = train_data

pred_train = np.array
train_data.shape
split_size = 10000
repeat_num = train_data.shape[0]//split_size +1
for num in range(repeat_num):

    slice_data = subset_data.iloc[num*split_size:(num+1)*split_size]
    print(slice_data.shape)

    subs = siamese_net.evaluate_SiamNet(
        input_finger6=slice_data['PTFP'].tolist(),
        input_finger2=slice_data['FCFP'].tolist(),
        input_finger3=slice_data['ECFP'].tolist(),
        input_finger4=slice_data['RDKIT'].tolist(),
        input_finger5=slice_data['TORS'].tolist(),
        input_finger1=slice_data['PAIR'].tolist(),
        # input_finger7=slice_data['MACCS'].tolist(),
        input_protein=slice_data['BERT'].tolist(),
        input_label=slice_data['dom_id'].tolist(),
        dropout=1.0,
        pre_trained_path='./model_save/kinase_6th',
        target_epoch=None,
    )
    print(subs.shape)
    pred_train = np.append(pred_train, subs)
    print(pred_train.shape)

pred_train = pred_train[1:].reshape(-1, 192)

pred_train.shape
pred = pred_train
# %%
train_data
train_df = pd.DataFrame(pred_train)
train_df['dom_id'] = subset_data['dom_id'].tolist()

train_df['smiles'] = subset_data['smiles'].tolist()
train_df.describe()

train_df.groupby('smiles').count().sort_values(1)

train_df.to_csv('./model_save/data/'+'train_data_bert.csv')



# %%

aa = train_df.groupby(by='dom_id').count()

aa[0].describe()


pred_train.shape
compound_vector = pred_train[:, :128].shape

train_df['dom_id'] = list(subset_data['dom_id'])
train_df['smiles'] = list(subset_data['smiles'])


# %% Train data clustering check ==============================================

from Bio.Blast import NCBIWWW
result_handle = NCBIWWW.qblast("blastn", "nt", "8332116")
