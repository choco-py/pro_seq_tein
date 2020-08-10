import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import os
import sys
print(tf.__version__)
print(os.getcwd())
os.chdir(
 '/home/hsohee/bert'
)
# os.chdir(
#     '/Users/soheehwang/bert/'
# )
#os.chdir(
#   '/mnt/d/sohee/KISS_siameseNet'
# )


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
# from src import data_loader as preprocessor

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
with open(data_check[2], 'rb') as f:
    train_data = pickle.load(f)

with open(data_check[1], 'rb') as f:
    test_data = pickle.load(f)

print('[shape]: train_data: '+ str(train_data.shape))
print('[shape]: test_data: '+ str(test_data.shape))
# test_data.dropna(axis=0, subset=['OUTPUT'], inplace=True)
# train_data.BERT.iloc[0].shape
# %% protein
# path = '/home/hsohee/git/pro_seq_tein/data/pred_data.pickle'
# with open(path, 'rb') as f:
#     protein_data = pickle.load(f)
#
# train_data.drop(columns=['seq', 'BERT'], inplace=True)
#
# train_data = train_data.join(protein_data.set_index('protein_name'), on='protein_name')
# with open('model_save/data/edited.pickle', 'wb') as f:
#     pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
#     print('Test Data Save.')
# # %%
# test_data.drop(columns=['seq', 'BERT'], inplace=True)
#
# test_data = test_data.join(protein_data.set_index('protein_name'), on='protein_name')
# with open('model_save/data/edited_test.pickle', 'wb') as f:
#     pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
#     print('Test Data Save.')

# %% Protein Similarity =======================================

with open(data_check[-1], 'rb') as f:
    protein_data = pickle.load(f)

from skbio.alignment import local_pairwise_align_protein
from skbio import Protein
from skbio.alignment import local_pairwise_align
# %%
seq_list = protein_data.iloc[:10].seq.tolist()
protein_num = len(seq_list)
similarity_matrix = np.zeros(shape=[protein_num, protein_num])

for i in range(len(seq_list)):
    for j in range(len(seq_list)):

        protein_similarity = local_pairwise_align_protein(
            seq1=Protein(seq_list[i]),
            seq2=Protein(seq_list[j]),
        )
        print(protein_similarity)

        similarity_matrix[i, j] = protein_similarity[1]
        print(similarity_matrix)
similarity_value = np.zeros(shape=similarity_matrix.shape)

for i in range(protein_num):
    for j in range(protein_num):
        value = (similarity_matrix[i, j]+similarity_matrix[j,i])/(similarity_matrix[i,i]+similarity_matrix[j,j])

        similarity_value[i, j] = value
        similarity_value[j, i] = value

        print(similarity_value)

rsk_family = protein_data[protein_data.protein_name.str.contains('RSK')]
protein_data.protein_name
protein_data.sort_values('protein_name', inplace=True)

protein_data['index'] = np.arange(488)
train_data.drop(columns=['index'], inplace=True)

train_data = train_data.join(protein_data.set_index('protein_name'), on='protein_name')

# %% Check JAK family ==========================================================
jak_list = jak_family.seq.tolist()
jak_num = len(jak_list)
jak_matrix = np.zeros(shape=[jak_num, jak_num])

for i in range(len(jak_list)):
    for j in range(len(jak_list)):

        protein_similarity = local_pairwise_align_protein(
            seq1=Protein(jak_list[i]),
            seq2=Protein(jak_list[j]),
        )
        print(protein_similarity)

        jak_matrix[i, j] = protein_similarity[1]
        print(jak_matrix)

jak_value = np.zeros(shape=jak_matrix.shape)

for i in range(protein_num):
    for j in range(protein_num):
        value = (jak_matrix[i, j]+jak_matrix[j,i])/(jak_matrix[i,i]+jak_matrix[j,j])

        jak_value[i, j] = value
        jak_value[j, i] = value

        print(jak_value)


similarity_value.mean()

jak_value.mean()


# %%

def calculate_sim(
    target_protein,
):

    protein_list = target_protein.seq.tolist()
    protein_num = len(protein_list)
    sim_matrix = np.zeros(shape=[protein_num, protein_num])
    print(f'==Start== with protein : {protein_num}')
    for i in range(len(protein_list)):
        for j in range(len(protein_list)):

            protein_similarity = local_pairwise_align_protein(
                seq1=Protein(protein_list[i]),
                seq2=Protein(protein_list[j]),
            )
            print(protein_similarity)

            sim_matrix[i, j] = protein_similarity[1]
            print(sim_matrix)

    sim_value = np.zeros(shape=sim_matrix.shape)

    for i in range(protein_num):
        for j in range(protein_num):
            value = (sim_matrix[i, j]+sim_matrix[j,i])/(sim_matrix[i,i]+sim_matrix[j,j])

            sim_value[i, j] = value
            sim_value[j, i] = value

            print(sim_value)

    return sim_matrix, sim_value
# %%
rsk_matrix, rsk_value = calculate_sim(target_protein=rsk_family)

rsk_value.mean()

num = np.random.randint(488, size=12)
random = protein_data.iloc[num]
random_matrix, random_value = calculate_sim(random)
random_value.mean()

num = np.random.randint(488, size=12)
random = protein_data.iloc[num]
random_matrix, random_value = calculate_sim(random)
random_value.mean()


rock_data = protein_data[protein_data.protein_name.str.contains('ROCK')]
rock_matrix, rock_value = calculate_sim(rock_data)
rock_value.mean()





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
# train_data.drop(columns=['seq', 'BERT'], inplace=True)

# train_data = train_data.join(protein_data.set_index('protein_name'), on='protein_name')
# train_data[train_data.RATIO.isna()==True]
# train_data
#
# test_data['RATIO_20'] = test_data['RATIO']
# test_data.drop(columns=['RATIO'], inplace=True)
# test_data = test_data.join(pro.set_index('protein_name'), on='protein_name')
# test_data[test_data.RATIO.isna()==True]
# %% Load blind kinase data ==================================================
# file_list = glob.glob('./data/blind_kinase/*.xlsx')
# blind_data = pd.read_excel(file_list[0])
# blind_data2 = pd.read_excel(file_list[-1])
#
# blind_data2 = preprocessor.fingerprint_converter(
#     input_data=blind_data2,
#     finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
#     smiles_cols='SMILES',
#     finger_size=1024,
# )
# blind_data = preprocessor.fingerprint_converter(
#     input_data=blind_data,
#     finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
#     smiles_cols='Smiles (Rdkit)',
#     finger_size=1024,
# )
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
    input_finger7_shape=(None, 166),
    input_protein_shape=(None, 488),
    buffer_size=10000000,
)
train_ok=False
# %% train
if train_ok:
    siamese_net.train_SiamNet(
       input_finger1=train_data['ECFP'].tolist(),
       input_finger2=train_data['FCFP'].tolist(),
       input_finger3=train_data['ECFP'].tolist(),
       input_finger4=train_data['RDKIT'].tolist(),
       input_finger5=train_data['TORS'].tolist(),
       input_finger6=train_data['PAIR'].tolist(),
       input_finger7=train_data['MACCS'].tolist(),
       input_protein=train_data['OUTPUT'].tolist(),
       input_label=train_data['dom_id'].tolist(),
       batch_size=128,
       epoch_num=10000,
       dropout=0.7,
       model_save_dir='./model_save/20200702',
       pre_trained_path='./model_save/20200702',
   )
print(os.getcwd())


len(train_data.OUTPUT.iloc[0])

# %% Test data evaluate ========================================================
subset_data = test_data.iloc[0:2]
pred = siamese_net.evaluate_SiamNet(
    input_finger1=subset_data['ECFP'].tolist(),
    input_finger2=subset_data['FCFP'].tolist(),
    input_finger3=subset_data['ECFP'].tolist(),
    input_finger4=subset_data['RDKIT'].tolist(),
    input_finger5=subset_data['TORS'].tolist(),
    input_finger6=subset_data['PAIR'].tolist(),
    input_finger7=subset_data['MACCS'].tolist(),
    input_protein=subset_data['OUTPUT'].tolist(),
    input_label=subset_data['dom_id'].tolist(),
    dropout=1.0,
    pre_trained_path='./model_save/model_save/20200702',
    target_epoch=None,
)


pred

 # %%
test_df = pd.DataFrame(pred)
test_df['dom_id'] = subset_data['dom_id'].tolist()
test_df['smiles'] = subset_data['smiles'].tolist()
pred.shape


test_df.to_csv('./model_save/data/'+'test_data_0702.csv')
test_df.groupby('smiles').count().sort_values(0)
aa = 'CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl'
test_df[test_df.smiles==aa]

test_df[test_df.smiles==aa]
# %% Train data evaluate =======================================================
subset_data = train_data

pred_train = np.array
train_data.shape
split_size = 10000
repeat_num = train_data.shape[0]//split_size +1
for num in range(repeat_num):

    slice_data = subset_data.iloc[num*split_size:(num+1)*split_size]
    print(slice_data.shape)

    subs = siamese_net.evaluate_SiamNet(
        input_finger1=slice_data['ECFP'].tolist(),
        input_finger2=slice_data['FCFP'].tolist(),
        input_finger3=slice_data['ECFP'].tolist(),
        input_finger4=slice_data['RDKIT'].tolist(),
        input_finger5=slice_data['TORS'].tolist(),
        input_finger6=slice_data['PAIR'].tolist(),
        input_finger7=slice_data['MACCS'].tolist(),
        input_protein=slice_data['OUTPUT'].tolist(),
        input_label=slice_data['dom_id'].tolist(),
        dropout=1.0,
        pre_trained_path='./model_save/20200702',
        target_epoch=None,
    )
    print(subs.shape)
    pred_train = np.append(pred_train, subs)
    print(pred_train.shape)

pred_train = pred_train.reshape(-1,160)

# subs = pred_train

# %%
train_data
train_df = pd.DataFrame(pred_train)
train_df['dom_id'] = subset_data['dom_id'].tolist()

train_df['smiles'] = subset_data['smiles'].tolist()
train_df.describe()

train_df.groupby('smiles').count().sort_values(1)

train_df.to_csv('./model_save/data/'+'train_data_0702.csv')

# %% Blind data evaluate =======================================================

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
    pre_trained_path='./model_save/kinase_2nd',
    target_epoch=None,
)
pred_blind.shape
blind_df = pd.DataFrame(pred_blind)
blind_df['dom_id'] = subset_data['Target priority'].tolist()
blind_df['smiles'] = subset_data['Smiles (Rdkit)'].tolist()
blind_df.describe()

blind_df.to_csv('./data/blind_kinase/'+'JAK_blind_2nd.csv')
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
    pre_trained_path='./model_save/kinase_2nd',
    target_epoch=None,
)
pred_blind.shape
blind_df = pd.DataFrame(pred_blind)
blind_df['dom_id'] = subset_data['Selectivity'].tolist()
blind_df['smiles'] = subset_data['SMILES'].tolist()
blind_df.describe()

blind_df.to_csv('./data/blind_kinase/'+'ROCK_blind_2nd.csv')
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

aa = train_df.groupby(by='dom_id').count()

aa[0].describe()


pred_train.shape
compound_vector = pred_train[:, :128].shape

train_df['dom_id'] = list(subset_data['dom_id'])
train_df['smiles'] = list(subset_data['smiles'])


tt =np.random.normal(0, 0.1, 488*488)
similarity_matrix = tt.reshape(488,488)



# %% ================= Similarity Version ======================================
# %% siameseSim: Build -------------------------------------------------------
os.getcwd()

glob.glob('model_save/kinase_3rd/last_weights/*')

import importlib
# from workspaces.src.anogan import anogan_sohee_origin as ANOGAN
from src import siamese_similarity as SiameseSim
importlib.reload(SiameseSim)
SiameseSim = SiameseSim.SiameseSim

tf.reset_default_graph()
siamese_sim = SiameseSim(

    input_finger1_shape=(None, 1024),
    input_finger2_shape=(None, 1024),
    input_finger3_shape=(None, 1024),
    input_finger4_shape=(None, 1024),
    input_finger5_shape=(None, 1024),
    input_finger6_shape=(None, 1024),
    input_finger7_shape=(None, 166),
    input_similarity_shape=(488, 488),
    buffer_size=10000000,
)
train_ok=True
# %% train
similarity_matrix.shape
subset =train_data
if train_ok:
    siamese_sim.train_SiamNet(
       input_finger1=subset['ECFP'].tolist(),
       input_finger2=subset['FCFP'].tolist(),
       input_finger3=subset['ECFP'].tolist(),
       input_finger4=subset['RDKIT'].tolist(),
       input_finger5=subset['TORS'].tolist(),
       input_finger6=subset['PAIR'].tolist(),
       input_finger7=subset['MACCS'].tolist(),
       input_label=subset['index'].tolist(),
       input_similarity=similarity_matrix,
       batch_size=128,
       epoch_num=10000,
       dropout=0.7,
       model_save_dir='./model_save/test',
       pre_trained_path=None,#'./model_save/reuse',
   )
print(os.getcwd())


# %%
with tf.Session() as sess:


    t11 = [[1], [2], [3]]#, 4, 5, 6]
    t21 = [[7], [8], [9]]#, 10, 11, 12]

    t1 = tf.placeholder(
        tf.float32,
        [3,1],
    )
    t2 = tf.placeholder(
        tf.float32,
        [3,1],
    )

    result = tf.concat([t1, t2], -1)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    aa = sess.run(result, feed_dict={t1: t11, t2:t21})
# tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
result

aa
