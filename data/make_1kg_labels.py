#!/bin/python
import numpy as np
import pandas as pd


## Assigns ground truth labels for (non-admixed) input genotypes (from 1KG)


# load input genotypes
z=np.load('/home/magu/deepmix/data/ALL_DNA_dataset/chm21_ALL_X.npz')
S=z['S'] # samples
V=z['V'] # variants


# load 1kg sample info
lab=pd.read_table('../labels/igsr_samples_cleaned_version.tsv', 
                   index_col='Sample name', usecols=['Sample name','Superpopulation code'])


# convert
pops=['AMR','AFR','EAS','EUR','SAS']
L=np.zeros(shape=(S.shape[0], V.shape[0], len(pops)), dtype=bool)
for i,s in enumerate(S): 
    p=pops.index(lab.loc[s[:-3]][0])
    L[i,:,p]=1


# save
np.savez_compressed('/home/magu/deepmix/data/ALL_DNA_dataset/chm21.label', L=L, S=S, V=V)
