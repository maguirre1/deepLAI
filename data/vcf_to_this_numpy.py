#!/bin/python
import sys
import gzip
import numpy as np

_README_="""
A script to convert genotypes from a vcf file to a compressed numpy (npz) 
file, as specified by another compressed numpy file. 

See vcf_to_numpy.py for more details. 

Author: Matthew Aguirre (magu[at]stanford[dot]edu)
"""

# parse input/output, ensure usage
if len(sys.argv) > 3:
    in_vcf=sys.argv[1]
    in_npz=sys.argv[2]
    out_prefix=sys.argv[3]
else:
    print("usage: python "+sys.argv[0]+" input.vcf input.npz output_prefix")
    sys.exit(4)

# load variant info from input npz
k=np.load(in_npz)['G'].shape[-1] # max number of variants
V=np.load(in_npz)['V'] # columns are chr pos ref alts
in_V=V.copy()

# recode input variants as chr_pos_ref:[ix,ref,alt1,...]
V={'_'.join(V[i,:3]):[i,V[i,2]]+V[i,3].split(',') for i in range(in_V.shape[0])}

# helper
def easy_open(f, mode):
    if f[-2:]=='gz':
        return gzip.open(f, mode)
    else:
        return open(f, mode)

# genotype data
G={i:None for i in range(in_V.shape[0])}
with easy_open(in_vcf, 'r') as f:
    for q,line in enumerate(map(lambda s:s.decode() if isinstance(s,bytes) else s, f)):
        if line[:2]=='##':
            continue
        elif line[0]=='#':
            S=np.array(line.rstrip().split()[9:])
        else:
            row=line.rstrip().split()
            vid='_'.join(row[:2]+[row[3]]) # variant id for this row: chr_pos_ref
            if vid in V:
                # map alleles
                npz_a=V[vid][1:] # npz alleles: [ref, alt1,...]
                vcf_a=[row[3]]+row[4].split(',')
                while len(vcf_a) < k:
                    vcf_a.append(None)
                vcf2npx={i:npz_a.index(v) if v in npz_a else None for i,v in enumerate(vcf_a)}
                # now take care of genotypes
                genos=row[9:]
                s1=np.array([[int(g.split('|')[0])==vcf2npx[i] for i in range(k)] for g in genos], dtype='bool')
                s2=np.array([[int(g.split('|')[1])==vcf2npx[i] for i in range(k)] for g in genos], dtype='bool')
                G[V[vid][0]]=np.vstack([s1,s2])

# back-fill missing values
for i in G:
    if G[i] is None:
        G[i]=np.zeros((2*S.shape[0],k), dtype='bool')

# recode and write to file
G=np.array([G[i] for i in range(in_V.shape[0])]).transpose((1,0,2))
print('S.shape='+str(S.shape))
print('V.shape='+str(in_V.shape))
print('G.shape='+str(G.shape))

# rename samples with strands
S=np.array([i+'_'+s for s in ['S1','S2'] for i in S])
print('2n={}'.format(S.shape[0]))

# write to file
np.savez_compressed(out_prefix+'.npz', G=G, V=in_V, S=S)
