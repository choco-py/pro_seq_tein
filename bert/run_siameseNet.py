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
    '/home/hsohee/bert/'
)
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
with open(data_check[-1], 'rb') as f:
    train_data = pickle.load(f)

with open(data_check[-3], 'rb') as f:
    test_data = pickle.load(f)

path = 'model_save/data/total_similarity.pickle'
with open(path, 'rb') as f:
    similarity_data = pickle.load(f)

print('[shape]: train_data: '+ str(train_data.shape))
print('[shape]: test_data: '+ str(test_data.shape))
print('[shape]: similarity_data: '+ str(similarity_data.shape))
# similarity_data_ = similarity_data
last = np.zeros(similarity_data.shape)
for i in range(similarity_data.shape[0]):
    for j in range(similarity_data.shape[0]):
        last[i,j] = (similarity_data[i,j]*similarity_data[j,i])/(similarity_data[i,i]*similarity_data[j,j])
# train_data.BERT.iloc[0].shape
pppp = 1-last



# %% protein
# path = '/home/hsohee/git/pro_seq_tein/data/pred_data.pickle'
# with open(path, 'rb') as f:
#     protein_data = pickle.load(f)
#
# train_data.drop(columns=['BERT', 'BERT_'], inplace=True)
# test_data.drop(columns=[ 'BERT_'], inplace=True)
# with open('model_save/data/train_data.pickle', 'wb') as f:
#     pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
#     print('Test Data Save.')
# with open('model_save/data/test_data.pickle', 'wb') as f:
#     pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

# #
# train_data = train_data.join(protein_data.set_index('protein_name'), on='protein_name')
# test_data = test_data.join(protein_data.set_index('protein_name'), on='protein_name')
#
# with open('model_save/data/edited.pickle', 'wb') as f:
#     pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
#     print('Test Data Save.')
# with open('model_save/data/edited_test.pickle', 'wb') as f:
#     pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

# %% Protein Similarity =======================================
#
# import swalign
#
# # choose your own values here… 2 and -1 are common.
# match = 2
# mismatch = -1
# scoring = swalign.NucleotideScoringMatrix(match, mismatch)
# swalign.ScoringMatrix('BLOSUM100')
# sw = swalign.LocalAlignment('BLOSUM100')  # you can also choose gap penalties, etc...
# alignment = swalign('ACACACTA','AGCACACA', ScoringMatrixValue='BLOSUM100')
# alignment.dump()
#
# seq_list = train_data.iloc[0:10].seq.tolist()
# train_data.
#
# for _ in range(10):
#
#     alignment = sw.align(seq_list[_], seq_list[0])
#     print(alignment.dump())
#
#
#
# from Bio.SubsMat import MatrixInfo as matlist
# matrix = matlist.blosum100
# from Bio import pairwise2
# from Bio.pairwise2 import format_alignment
# # %%
# # pairwise2.align.globaldx("KEVLA", "EVL", scoring)
# for _ in range(4):
#
#     alignment = sw.align(seq_list[_], seq_list[0])
#     print(alignment.dump())
#     # for a in pairwise2.align.localdx(seq_list[_], seq_list[0], matrix, one_alignment_only=True):
#     #     print(format_alignment(*a))
#     # gg = aligner.align(seq_list[_], seq_list[0])
#     # g = gg[0]
#     # print(g)
#     # print(g.score)
#     tt = local_pairwise_align_protein(seq1=Protein(seq_list[_]), seq2=Protein(seq_list[0]), gap_open_penalty=11, gap_extend_penalty=1)
#     print(tt)
#
# from Bio import Align
#
#
# aligner = Align.PairwiseAligner()
# aligner.substitution_matrix = matrix
# alignments = aligner.align("KEVLA", "EVL")
# alignments = list(alignments)
# print("Number of alignments: %d" % len(alignments))
# alignment = alignments[0]
# print("Score = %.1f" % alignment.score)
#
# print(alignment)
#
# from skbio.alignment import local_pairwise_align_protein
# from skbio import Protein
# from skbio.alignment import local_pairwise_align
# def align(sequence1, sequence2, gap_penalty, substitution_matrix, local):
#     if local:
#         return local_pairwise_align(sequence1, sequence2, gap_penalty, gap_penalty, substitution_matrix)
#     else:
#         return global_pairwise_align(sequence1, sequence2, gap_penalty, gap_penalty, substitution_matrix)
#
#
#
# aln, score, _ = align('HEAGAWGHEE', 'PAWHEAE',)# 8, 'blosum50', False)
#
# protein_list
# seq_list = train_data.seq.tolist()
# similarity_matrix = np.zeros(shape=[len(seq_list), len(seq_list)])
#
# # %%
# for i in range(len(seq_list)):
#     for j in range(len(seq_list)):
#
#         protein_similarity = local_pairwise_align_protein(
#             seq1=Protein(seq_list[i]),
#             seq2=Protein(seq_list[j]),
#         )
#         print(protein_similarity)
#
#         similarity_matrix[i, j] = protein_similarity[1]
#         print(similarity_matrix)
#
# local_pairwise_align_protein(
#     seq1=Protein(seq_list[3]),
#     seq2=Protein(seq_list[0]),
# )
#
# local_pairwise_align_protein(
#     seq1=Protein(seq_list[0]),
#     seq2=Protein(seq_list[3]),
# )
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
# os.getcwd()
#
# glob.glob('model_save/kinase_3rd/last_weights/*')
#
# import importlib
# # from workspaces.src.anogan import anogan_sohee_origin as ANOGAN
# from src import siamese_network as SiameseNet
# importlib.reload(SiameseNet)
# SiameseNet = SiameseNet.SiameseNet
#
# tf.reset_default_graph()
# siamese_net = SiameseNet(
#
#     input_finger1_shape=(None, 1024),
#     input_finger2_shape=(None, 1024),
#     input_finger3_shape=(None, 1024),
#     input_finger4_shape=(None, 1024),
#     input_finger5_shape=(None, 1024),
#     input_finger6_shape=(None, 1024),
#     input_finger7_shape=(None, 166),
#     input_protein_shape=(None, 488),
#     buffer_size=10000000,
# )
# train_ok=False
# # %% train
# if train_ok:
#     siamese_net.train_SiamNet(
#        input_finger1=train_data['ECFP'].tolist(),
#        input_finger2=train_data['FCFP'].tolist(),
#        input_finger3=train_data['ECFP'].tolist(),
#        input_finger4=train_data['RDKIT'].tolist(),
#        input_finger5=train_data['TORS'].tolist(),
#        input_finger6=train_data['PAIR'].tolist(),
#        input_finger7=train_data['MACCS'].tolist(),
#        input_protein=train_data['OUTPUT'].tolist(),
#        input_label=train_data['dom_id'].tolist(),
#        batch_size=128,
#        epoch_num=10000,
#        dropout=0.7,
#        model_save_dir='./model_save/20200702',
#        pre_trained_path='./model_save/20200702',
#    )
# print(os.getcwd())
#
#
# len(train_data.OUTPUT.iloc[0])
#
# # %%
# pred = siamese_net.evaluate_SiamNet(
#        input_finger1=test_data['ECFP'].tolist(),
#        input_finger2=test_data['FCFP'].tolist(),
#        input_finger3=test_data['ECFP'].tolist(),
#        input_finger4=test_data['RDKIT'].tolist(),
#        input_finger5=test_data['TORS'].tolist(),
#        input_finger6=test_data['PAIR'].tolist(),
#        input_finger7=test_data['MACCS'].tolist(),
#        input_protein=test_data['OUTPUT'].tolist(),
#        input_label=test_data['dom_id'].tolist(),
#     dropout=1.0,
#     pre_trained_path='./model_save/20200702',
#     target_epoch=None,
# )
#
#
#
#
# # %%
# test_df = pd.DataFrame(pred)
# test_df['dom_id'] = subset_data['dom_id'].tolist()
# test_df['smiles'] = subset_data['smiles'].tolist()
# pred.shape
#
#
# test_df.to_csv('./model_save/data/'+'test_data.csv')
# test_df.groupby('smiles').count().sort_values(0)
# aa = 'CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl'
# test_df[test_df.smiles==aa]
#
# test_df[test_df.smiles==aa]
# # %%
#
# subset_data = blind_data
# skl_len = blind_data.shape[0]
#
# pred_blind = siamese_net.evaluate_SiamNet(
#     input_finger1=subset_data['PTFP'].tolist(),
#     input_finger2=subset_data['FCFP'].tolist(),
#     input_finger3=subset_data['ECFP'].tolist(),
#     input_finger4=subset_data['RDKIT'].tolist(),
#     input_finger5=subset_data['TORS'].tolist(),
#     input_finger6=subset_data['PAIR'].tolist(),
#     input_finger7=subset_data['MACCS'].tolist(),
#     input_protein=test_data['RATIO'].iloc[:skl_len].tolist(),
#     input_label=subset_data.index.tolist(),
#     dropout=1.0,
#     pre_trained_path='./model_save/kinase_2nd',
#     target_epoch=None,
# )
# pred_blind.shape
# blind_df = pd.DataFrame(pred_blind)
# blind_df['dom_id'] = subset_data['Target priority'].tolist()
# blind_df['smiles'] = subset_data['Smiles (Rdkit)'].tolist()
# blind_df.describe()
#
# blind_df.to_csv('./data/blind_kinase/'+'JAK_blind_2nd.csv')
# subset_data['PTFP'].shape
#
# # %%
# subset_data = blind_data2
# skl_len = blind_data2.shape[0]
#
# pred_blind = siamese_net.evaluate_SiamNet(
#     input_finger1=subset_data['PTFP'].tolist(),
#     input_finger2=subset_data['FCFP'].tolist(),
#     input_finger3=subset_data['ECFP'].tolist(),
#     input_finger4=subset_data['RDKIT'].tolist(),
#     input_finger5=subset_data['TORS'].tolist(),
#     input_finger6=subset_data['PAIR'].tolist(),
#     input_finger7=subset_data['MACCS'].tolist(),
#     input_protein=test_data['RATIO'].iloc[:skl_len].tolist(),
#     input_label=subset_data.index.tolist(),
#     dropout=1.0,
#     pre_trained_path='./model_save/kinase_2nd',
#     target_epoch=None,
# )
# pred_blind.shape
# blind_df = pd.DataFrame(pred_blind)
# blind_df['dom_id'] = subset_data['Selectivity'].tolist()
# blind_df['smiles'] = subset_data['SMILES'].tolist()
# blind_df.describe()
#
# blind_df.to_csv('./data/blind_kinase/'+'ROCK_blind_2nd.csv')
# subset_data['PTFP'].shape
# # %%
#
# subset_data = skl_data
# skl_len = skl_data.shape[0]
#
# pred_skl = siamese_net.evaluate_SiamNet(
#     input_finger1=list(subset_data['PTFP']),
#     input_finger2=list(subset_data['FCFP']),
#     input_finger3=list(subset_data['ECFP']),
#     input_finger4=list(subset_data['RDKIT']),
#     input_finger5=list(subset_data['TORS']),
#     input_finger6=list(subset_data['PAIR']),
#     input_finger7=list(subset_data['MACCS']),
#     input_protein=list(test_data['RATIO'].iloc[:skl_len]),
#     input_label=list(subset_data['Name']),
#     dropout=1.0,
#     pre_trained_path='./model_save/kinase_2nd',
#     target_epoch=None,
# )
# pred_skl.shape
# skl_df = pd.DataFrame(pred_skl)
# skl_df['dom_id'] = list(subset_data['Category'])
# skl_df['smiles'] = list(subset_data['Canonical_Smiles'])
# skl_df.describe()
#
# skl_df.to_csv('./model_save/data/'+'skl_data_2nd.csv')
#
# # %%
# subset_data = train_data
#
# pred_train = np.array
# train_data.shape
# split_size = 10000
# repeat_num = train_data.shape[0]//split_size +1
# for num in range(repeat_num):
#
#     slice_data = subset_data.iloc[num*split_size:(num+1)*split_size]
#     print(slice_data.shape)
#
#     subs = siamese_net.evaluate_SiamNet(
#         input_finger1=slice_data['PTFP'].tolist(),
#         input_finger2=slice_data['FCFP'].tolist(),
#         input_finger3=slice_data['ECFP'].tolist(),
#         input_finger4=slice_data['RDKIT'].tolist(),
#         input_finger5=slice_data['TORS'].tolist(),
#         input_finger6=slice_data['PAIR'].tolist(),
#         input_finger7=slice_data['MACCS'].tolist(),
#         input_protein=slice_data['RATIO'].tolist(),
#         input_label=slice_data['dom_id'].tolist(),
#         dropout=1.0,
#         pre_trained_path='./model_save/kinase_2nd',
#         target_epoch=None,
#     )
#     print(subs.shape)
#     pred_train = np.append(pred_train, subs)
#     print(pred_train.shape)
#
# pred_train = pred_train[1:].reshape(-1, 148)
#
#
#
# # %%
# train_data
# train_df = pd.DataFrame(pred_train)
# train_df['dom_id'] = subset_data['dom_id'].tolist()
#
# train_df['smiles'] = subset_data['smiles'].tolist()
# train_df.describe()
#
# train_df.groupby('smiles').count().sort_values(1)
#
# train_df.to_csv('./model_save/data/'+'train_data_2nd.csv')
#
#
#
# # %%
#
# aa = train_df.groupby(by='dom_id').count()
#
# aa[0].describe()
#
#
# pred_train.shape
# compound_vector = pred_train[:, :128].shape
#
# train_df['dom_id'] = list(subset_data['dom_id'])
# train_df['smiles'] = list(subset_data['smiles'])
#

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
    # input_protein_shape=(None, 488),
    buffer_size=10000000,
)
train_ok=True
# %% train
if train_ok:
    siamese_sim.train_SiamNet(
       input_finger1=train_data['ECFP'].tolist(),
       input_finger2=train_data['FCFP'].tolist(),
       input_finger3=train_data['ECFP'].tolist(),
       input_finger4=train_data['RDKIT'].tolist(),
       input_finger5=train_data['TORS'].tolist(),
       input_finger6=train_data['PAIR'].tolist(),
       input_finger7=train_data['MACCS'].tolist(),
       input_similarity=1-last,
       input_label=train_data['LABEL'].tolist(),
       batch_size=128,
       epoch_num=10000,
       dropout=0.7,
       model_save_dir='./model_save/similarity_minus1',
       pre_trained_path='./model_save/similarity_minus1',#'./model_save/reuse',
   )
print(os.getcwd())
# %%
# a = np.arange(100).reshape(-1,50)
# a.shape
# b = np.arange(100).reshape(-1,50)
# similar = np.arange(100*100).reshape(-1,100)
# a.reshape(-1,1)
#
# X = tf.placeholder(
#     tf.int32,
#     [None, 50],
# )
#
# Y = tf.placeholder(
#     tf.int32,
#     [None, 50],
# )
# reX =tf.reshape(X, [-1, 1])
# reY =tf.reshape(Y, [-1, 1])
#
# sim = tf.placeholder(
#     tf.float32,
#     [100,100],
# )
# result = tf.concat([reX,reY],axis=1)
# reshape = tf.reshape(result, [-1, 2])
# similarity_value =tf.gather_nd(
#     params=sim,
#     indices=[result],
#     # batch_dims=self.BATCH_SIZE,
# )
# with tf.Session() as sess:
#
#     t, y ,a, value = sess.run([result, reshape, X, similarity_value], feed_dict={X:a, Y:b, sim:similar})
#     print(t)
# t[1]
# value.shape
# y.shape
