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
    seq_cols='seq',
    protein_path=None,
    protein_type='RATIO',
):
    if seq_cols:
        protein_seq = input_data[seq_cols]
        print('Protein cols <%s> is Completed.' % seq_cols)

    else:
        fasta_list = glob.glob(protein_path+'*.fasta')
        protein_name = list(
            map(os.path.basename, fasta_list)
        )

        protein_seq = list(
            map(
                lambda x: AAC.CalculateAADipeptideComposition(SeqIO.read(x, 'fasta').seq).values(),
                fasta_list,
            )
        )

        protein_table = pd.DataFrame(columns=['Category', 'protein_seq'])

        protein_table['Category'] = protein_name
        protein_table['Category'] = protein_table['Category'].str.split('.').apply(
            lambda x: x[0]).str.replace('-', '').str.upper()

        print(protein_seq)
        protein_ratio = list(
            map(
                lambda x: list(AAC.CalculateAADipeptideComposition(x).values()),
                protein_seq,
            )
        )
# %% protein_preprocess

        protein_ratio = list(
            map(
                lambda x: list(AAC.CalculateAADipeptideComposition(x).values()),
                protein_table['seq'],
            )
        )

        protein_table['protein_seq'] = protein_seq
        input_data = input_data.join(protein_table.set_index('Category'), on='Category')



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

        protein_table = input_data[[protein_cols, seq_cols]]
        protein_table.drop_duplicates(protein_cols, inplace=True)

        protein_ratio = list(
            map(
                lambda x: list(AAC.CalculateAADipeptideComposition(x).values()),
                protein_table[seq_cols],
            )
        )

        protein_table['RATIO'] = protein_ratio
        protein_table.drop(columns=seq_cols, inplace=True)
        input_data = input_data.join(protein_table.set_index(protein_cols), on=protein_cols)

        # input_data[protein_type] = protein_ratio

    elif protein_type=='RATIO_reverse':

        protein_table = input_data[[protein_cols, seq_cols]]
        protein_table.drop_duplicates(protein_cols, inplace=True)
        protein_table['seq_reverse'] = protein_table[seq_cols].apply(lambda x: x[::-1])

        protein_ratio_ = list(
            map(
                lambda x: list(AAC.CalculateAADipeptideComposition(x).values()),
                protein_table['seq_reverse'],
            )
        )
        protein_table.drop(columns=[seq_cols, 'seq_reverse'], inplace=True)
        protein_table[protein_type] = protein_ratio_
        input_data = input_data.join(protein_table.set_index(protein_cols), on=protein_cols)

        return input_data
        # input_data[protein_type] = protein_ratio_


        print('========='+protein_type+"=========")




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
