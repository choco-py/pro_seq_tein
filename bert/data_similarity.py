import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

os.chdir(
    '/home/hsohee/bert/'
)

from skbio.alignment import local_pairwise_align_protein
from skbio import Protein
from skbio.alignment import local_pairwise_align

# %%
path = '/home/hsohee/git/pro_seq_tein/data/pred_data.pickle'
with open(path, 'rb') as f:
    protein_data = pickle.load(f)


seq_list = protein_data.seq.tolist()
similarity_matrix = np.zeros(shape=[len(seq_list), len(seq_list)])
preprocess = False
if preprocess:
    for i in range(len(seq_list)):
        for j in range(len(seq_list)):

            protein_similarity = local_pairwise_align_protein(
                seq1=Protein(seq_list[i]),
                seq2=Protein(seq_list[j]),
            )

            similarity_matrix[i, j] = protein_similarity[1]
            print(i, j)


        with open('model_save/data/similarity_matrix.pickle', 'wb') as f:
            pickle.dump(similarity_matrix, f, pickle.HIGHEST_PROTOCOL)

            print(f'{i}th protein similarity matrix Save.')
# %% open

protein_data['LABEL'] = np.arange(488)
# with open(path, 'wb') as f:
#         pickle.dump(protein_data, f, pickle.HIGHEST_PROTOCOL)


# %%

path = 'model_save/data/total_similarity.pickle'
with open(path, 'rb') as f:
    similarity_data = pickle.load(f)
print('[shape]: similarity_data: '+ str(similarity_data.shape))
last = np.zeros(similarity_data.shape)
for i in range(similarity_data.shape[0]):
    for j in range(similarity_data.shape[0]):
        last[i,j] = (similarity_data[i,j]*similarity_data[j,i])/(similarity_data[i,i]*similarity_data[j,j])
# train_data.BERT.iloc[0].shape
print('check cacul')
print(last)
last.mean()
fig = plt.figure(figsize=(10,3))
plt.hist(last)
