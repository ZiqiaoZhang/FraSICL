from TrainingFramework.ChemUtils import *
from TrainingFramework.FileUtils import *
import os

# original dataset file
dataset_file = '/remote-home/zqzhang/Data/PretrainData/pubchem-10m-clean.txt'

fileloader = FileLoader(dataset_file)
dataset = fileloader.load()
print(f"len(datast):{len(dataset)}")

# generate the 10M Large dataset
if not os.path.exists("/remote-home/zqzhang/Data/PretrainData/pubchem-10m-screened.txt"):
    print("generating 10M dataset....")
    discard_cnt = 0
    screened_cnt = 0
    with open("/remote-home/zqzhang/Data/PretrainData/pubchem-10m-screened.txt",'w') as f:
        for smiles in dataset:
            mol = GetMol(smiles)    # check mol
            if mol:
                atom_num = GetAtomNum(mol)
                singlebond_num = len(GetSingleBonds(mol))
                if (atom_num < 100) & (singlebond_num>0):   # check max atom_num and check breakable
                    mol_error_flag = 0
                    for atom in mol.GetAtoms():
                        if atom.GetDegree() > 5:            # check atom degree
                            mol_error_flag = 1
                            break
                    if not mol_error_flag:
                        screened_cnt += 1
                        print(f"obtain mol idx: {screened_cnt}")
                        f.write(smiles)
                    else:
                        discard_cnt += 1
                else:
                    discard_cnt += 1

    print(f"screened count:{screened_cnt}")
    print(f'discarded smiles num: {discard_cnt}')


# generate 200K small set (randomly)

import random
random.seed(8)
random.shuffle(dataset)
if not os.path.exists("/remote-home/zqzhang/Data/PretrainData/pubchem-200K-screened.txt"):
    print(f"generating 200K dataset....")
    discard_cnt = 0
    screened_cnt = 0
    with open("/remote-home/zqzhang/Data/PretrainData/pubchem-200K-screened.txt", 'w') as f:
        for smiles in dataset:
            mol = GetMol(smiles)
            if mol:
                atom_num = GetAtomNum(mol)
                singlebond_num = len(GetSingleBonds(mol))
                if (atom_num < 100) & (singlebond_num>0):
                    mol_error_flag = 0
                    for atom in mol.GetAtoms():
                        if atom.GetDegree() > 5:
                            mol_error_flag = 1
                            break
                    if not mol_error_flag:
                        screened_cnt += 1
                        print(f"obtain mol idx: {screened_cnt}")
                        f.write(smiles)
                    else:
                        discard_cnt += 1
                else:
                    discard_cnt += 1
            if screened_cnt == 200000:
                break
    print(f"screened count:{screened_cnt}")
    print(f"discarded smiles num:{discard_cnt}")