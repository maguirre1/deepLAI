#!/bin/python
import numpy as np
import pandas as pd


## Assigns ground truth labels for (non-admixed) input genotypes (from 1KG)


# paths
info='reference_panel_metadata_w_qs.tsv'
in_S='/home/magu/deepmix/data/reference_panel/unzipped/panel_chr9.S.npy'
in_V='/home/magu/deepmix/data/reference_panel/unzipped/panel_chr9.V.npy'
out_L='/home/magu/deepmix/data/reference_panel/labels/panel_chr9.L'


# load input genotypes
S=np.load(in_S)
V=np.load(in_V)


# load sample info
lab=pd.read_table(info, usecols=['Sample','Panel'], index_col='Sample')


# convert
pops=list(sorted(lab['Panel'].value_counts().index.tolist()))
pops.remove('AHG')
lab=lab[lab['Panel']!='AHG']
L=np.zeros(shape=(S.shape[0], V.shape[0], len(pops)), dtype=bool)
for i,s in enumerate(S): 
    # these aren't in the tsv -- by default they're in train set
    if s[:-3]=='SS6004478' or s[:-3]=='SS6004477':
        p=pops.index('OCE')
    elif s[:-3] in lab.index:
        p=pops.index(lab.loc[s[:-3]][0])
    L[i,:,p]=1


# save
np.savez_compressed(out_L, L=L, S=S, V=V)
