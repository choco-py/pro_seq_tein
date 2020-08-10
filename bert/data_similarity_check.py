import os

import pickle
import pandas as pd
import numpy as np

os.chdir(
    '/home/hsohee/bert/'
)

from skbio.alignment import local_pairwise_align_protein
from skbio import Protein
from skbio.alignment import local_pairwise_align


path1 = './model_save/data/similarity_matrix_reverse.pickle'
with open(path1, 'rb') as f:
    similarity_matrix_reverse = pickle.load(f)

path2 = './model_save/data/similarity_matrix.pickle'
with open(path2, 'rb') as f:
    similarity_matrix = pickle.load(f)

num = 281
total = np.append(similarity_matrix[:num], similarity_matrix_reverse[num:])
total = total.reshape(-1, 488)

with open('model_save/data/total_similarity.pickle', 'wb') as f:
    pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)

    print('Total protein similarity matrix Save.')


# print(similarity_matrix)
# print(total.shape)
# print(total)
#

# seq_list = protein_data.seq.tolist()
# similarity_matrix = np.zeros(shape=[len(seq_list), len(seq_list)])
#
#
# for i in range(len(seq_list)):
#     for j in range(len(seq_list)):
#
#         protein_similarity = local_pairwise_align_protein(
#             seq1=Protein(seq_list[i]),
#             seq2=Protein(seq_list[j]),
#         )
#
#         similarity_matrix[i, j] = protein_similarity[1]
#         print(i, j)
#
#
#     with open('model_save/data/similarity_matrix.pickle', 'wb') as f:
#         pickle.dump(similarity_matrix, f, pickle.HIGHEST_PROTOCOL)
#
#         print(f'{i}th protein similarity matrix Save.')
