#!/bin/python3
import numpy as np
import pandas as pd


## Makes a train/dev/test split for a specified input (npz) genotype dataset


# load individuals and remove strand encoding
S=np.load('/home/magu/deepmix/data/ALL_DNA_dataset/chm21_ALL_X.npz')['S']
S=list(set([i[:-3] for i in list(S)])) # remove strand encoding "_S1", "_S2"


# get demographic features
D=pd.read_csv('igsr_samples_cleaned_version.tsv', sep='\t', index_col=0,
        usecols=['Sample name','Population code','Superpopulation code']).loc[S,:]


# remove these populations due to admixture
pops_to_remove=['ASW','MXL','CEU','PUR','PEL','CLM','ACB'] # +['GIH','ITU','STU']"
D=D[~D['Population code'].isin(pops_to_remove)]


# select 50 people (25 dev, 25 test) from each population
superpops=['AFR','EAS','EUR','SAS'] # note the above removes AMR
dev,test=[],[]
np.random.seed(212)
for pop in superpops:
    inds=np.random.choice(D.index[D['Superpopulation code']==pop].tolist(), size=50)
    dev+=list(inds[:25])
    test+=list(inds[25:])
train=[i for i in D.index.tolist() if i not in dev and i not in test]


# write to file -- one with strand labels and one without
root='/home/magu/deepmix/data/ALL_DNA_dataset/'
for name,obj in zip(['train','dev','test'], [train,dev,test]):
    with open(root+'chm21.'+name+'.txt', 'w') as o:
        o.write('\n'.join([a+s for s in ['_S1','_S2'] for a in obj])+'\n')
    with open(root+'chm21.'+name+'.inds.txt', 'w') as o:
        o.write('\n'.join([a for a in obj])+'\n')
    # and a label file
    D.loc[obj,'Superpopulation code'].to_csv(root+'chm21.'+name+'.superpop.txt', 
            sep='\t', header=None)
