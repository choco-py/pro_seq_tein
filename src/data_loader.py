import os
import random
import numpy as np
import math
import glob
import pandas as pd
from pprint import pprint as pp
from sklearn.model_selection import train_test_split
from propy import AAComposition as AAC

from collections import defaultdict
from Bio import SeqIO
from Bio import pairwise2

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as ECFP
from rdkit.Chem.rdmolops import PatternFingerprint as PTFP
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint as MACCS
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect as PAIR
from rdkit.Chem.rdMolDescriptors import  GetHashedTopologicalTorsionFingerprintAsBitVect as TORS
from rdkit.Chem import RDKFingerprint as RDKIT

# %%

def data_loader(
    data_path,
    ):

    data_list = glob.glob(data_path+'*.csv')

    print('Found %s Files.' % len(data_list))

    full_data = pd.concat(
        map(
            pd.read_csv,
            data_list,
        )
    )
    print(
        '[shape]: ', full_data.shape,
    )

    return full_data

def fingerprint_converter(
    input_data,
    finger_list,
    smiles_cols='CANONICAL_SMILES',
    finger_size=1024,
):
    mol_list = list(
        map(
            MolFromSmiles,
            input_data[smiles_cols],
        )
    )
    print('Mol-list Ready, ')


    transFunc_dict = {
        'PTFP': lambda x: list(PTFP(x, fpSize=finger_size)),
        'ECFP': lambda x: list(ECFP(x, 2, nBits=finger_size)),
        'FCFP': lambda x: list(ECFP(x, 2, useFeatures=True, nBits=finger_size)),
        'MACCS': lambda x: list(MACCS(x))[1:],
        'TORS': lambda x: list(TORS(x, nBits=finger_size).ToBitString()),
        'PAIR':lambda x: list(PAIR(x, nBits=finger_size).ToBitString()),
        'RDKIT': lambda x: list(RDKIT(x, fpSize=finger_size).ToBitString()),
    }

    for finger_type in finger_list:

        input_data[finger_type] = list(
            map(
            transFunc_dict[finger_type],
            mol_list,
            )
        )
        print('========= ['+finger_type+']'+' Converted! =========')

    return input_data

def protein_loader(
    input_data,
    protein_cols=None,
    protein_path=None,
    protein_type='RATIO',
):
    if protein_cols:
        protein_seq = input_data[protein_cols]
        print('Protein cols <%s> is Completed.' % protein_cols)

    else:
        fasta_list = glob.glob(protein_path+'*.fasta')
        fasta_list.sort()

        protein_seq = list(
            map(
                lambda x: SeqIO.read(x, 'fasta').seq,
                fasta_list,
            )
        )
        print('Protein fasta reader is Completed.')

    if protein_type=='VEC':
        word_dict = defaultdict(lambda: len(word_dict))

        string = '-' + protein_seq + '='

        protein_vec = list(
            map(
                lambda x: np.array([word_dict[x[i:i+1]] for i in range(len(x))]),
                string,
            )
        )
        input_data[protein_type] = protein_vec
        print('========='+protein_type+"=========")

    elif protein_type=='RATIO':

        protein_ratio = list(
            map(
                lambda x: list(AAC.CalculateAADipeptideComposition(x).values()),
                protein_seq,
            )
        )
        print('========='+protein_type+"=========")

        input_data[protein_type] = protein_ratio

    return input_data


def protein_similarity(
    protein_seq1,
    protein_seq2,

):
#     protein_seq1 = test_data[0:1]
#     protein_seq2 = test_data[1:2]

    alignments = pairwise2.align.globalxx(protein_seq1.seq.values, protein_seq2.seq.values)


def data_preprocessor(
    dataset,
    smiles_cols='CANONICAL_SMILES',
    finger_list=['PTFP', 'ECFP', 'FCFP', 'RDKIT', 'MACCS', 'TORS', 'PAIR'],
    protein_cols=None,
    protein_path=None,
    protein_type='RATIO',
    train_ratio=0.2,
):

    dataset = fingerprint_converter(
        dataset,
        finger_list,
        smiles_cols,
    )

    dataset = protein_loader(
        input_data=dataset,
        protein_cols=protein_cols,
        protein_path=None,
        protein_type=protein_type,
    )

    train, test = train_test_split(
            dataset,
            test_size=train_ratio,
    )
    print('Train / [shape]: '+ str(train.shape))
    print('Test  / [shape]: '+ str(test.shape))

    return train, test
