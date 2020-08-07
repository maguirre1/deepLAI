#!/bin/python3
import sys
import os
import tensorflow as tf
from segnet import segnet


ix=int(sys.argv[1])

# tensorflow courtesy options: use GPU1, don't take up all the memory
tf.config.experimental.set_visible_devices([], 'GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="1"
for d in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)


# hyperparameters
n_classes=8
width=[2,4,6,8,16,24,32,48]
n_filters=[4,6,8,12,16,32,64]
#pool_size=[2,3,4,6,8,16]
pool_size=[2]
n_blocks=[2,3,4,5,6,7,8]
pparams=[(a,b,c,d) for a in width for b in n_filters for c in pool_size for d in n_blocks]

# open log file
with open('temp/hparam_to_nparam.{}.tsv'.format(ix), 'w') as i:
    # header
    i.write('\t'.join(['width','n_filter','maxpool','depth','n_par\n']))
    # iterate through parameters of interest
    for j,(w,nf,ps,nb) in enumerate(pparams):
        if j==ix:
            nv=(2**20)-((2**20) % (ps**nb))
            model=segnet(input_shape=(nv,2), n_classes=n_classes, width=w, n_filters=nf, pool_size=ps, n_blocks=nb)
            n_par=model.count_params()
            i.write('\t'.join(map(str, [w, nf, ps, nb, n_par]))+'\n')

