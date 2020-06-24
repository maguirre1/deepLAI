#!/bin/python
import sys
import gzip,bgzip
import numpy as np

_README_="""
A script to convert genotypes from a vcf file to a compressed numpy (npz) 
file. Currently takes all variants in the file, regardless of filtration 
status (might be worth revisiting). 

Author: Matthew Aguirre (magu[at]stanford[dot]edu)
"""

# parse input/output, ensure usage
if len(sys.argv) > 2:
    in_vcf=sys.argv[1]
    out_prefix=sys.argv[2]
else:
    print("usage: python read_vcf.py input.vcf output_prefix")
    sys.exit(4)

# this is just a magic number -- max number of alleles (incl. ref)
k=2

# helper
def easy_open(f, mode):
    if f[-2:]=='gz':
        #try:
            #print("shit")
            #q=gzip.open(f, mode).readline()
            #print("fuck")
        return gzip.open(f, mode)
        #except SyntaxError:
        #    return gzip.open(f, mode+'b')
    else:
        return open(f, mode)

# genotype data
V,G=[],[]
with easy_open(in_vcf, 'r') as f:
    for q,line in enumerate(map(lambda s:s.decode(),f)):
        if q % 10000 == 0:
            print(q)
        if line[:2]=='##':
            continue
        elif line[0]=='#':
            S=np.array(line.rstrip().split()[9:])
        else:
            row=line.rstrip().split()
            V.append(np.array([row[0],row[1],row[3],row[4]]))
            row=row[9:]
            s1=np.array([[int(g.split('|')[0])==i for i in range(k)] for g in row], dtype='bool')
            s2=np.array([[int(g.split('|')[1])==i for i in range(k)] for g in row], dtype='bool')
            G.append(np.vstack([s1,s2]))

V=np.array(V)
G=np.array(G).transpose((1,0,2))
print('S.shape='+str(S.shape))
print('V.shape='+str(V.shape))
print('G.shape='+str(G.shape))

# rename samples with strands
S=np.array([i+'_'+s for s in ['S1','S2'] for i in S])
print('2n={}'.format(S.shape[0]))

# write to file
np.savez_compressed(out_prefix+'.npz', G=G, V=V, S=S)
