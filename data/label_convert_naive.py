#!/bin/python
import numpy as np
import pandas as pd
import sys
import os

## input: result from rfmix simulate
in_result=sys.argv[1]

V=np.load('/home/magu/deepmix/data/ALL_DNA_dataset/chm21_ALL_X.npz')['V']
X=pd.read_csv(in_result, sep='\t', header=0, index_col=None)
S=np.array([s.replace('.0','_S1').replace('.1','_S2') for s in X.columns[2:]])
L=X.values.T[2:,:]

np.savez_compressed(os.path.basename(in_result), S=S, V=V, L=L)

