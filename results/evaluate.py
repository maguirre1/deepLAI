#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
import scipy.stats as ss
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from segnet import segnet


# specify model, datasets
prefix='weights/chr20.full.conv3.168'
if len(sys.argv) > 1:
	dataset=sys.argv[1]
else:
	dataset="test_10gen.no_OCE_WAS"
in_x="/scratch/users/magu/deepmix/data/simulated_chr20/numpy/"+dataset+".query.npz"
in_y="/scratch/users/magu/deepmix/data/simulated_chr20/label/"+dataset+".result.npz"
print(in_x)
print(in_y)

# consider proper variants
v = np.loadtxt(prefix+'.var_index.txt', dtype=int)

# declare model, compile, load weights -- perhaps make this automated with the file?
model=segnet(input_shape=(v.shape[0], 2), n_classes=5, n_blocks=4, n_filters=16, width=16)
model.compile(tf.keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(prefix+'.h5')


# load data
anc=np.array(['AFR','EAS','EUR','NAT','SAS']) # fixed for all test datasets
x=np.load(in_x)
y=np.load(in_y)
X=x['G'][:,v,:]
S=x['S']
V=x['V'][v]
Y=y['L'][np.ix_(np.array([np.where(y['S']==s)[0][0] for s in S]),v)]-1
print(X.shape, Y.shape)


# preliminary evaluation
Yhat=model.predict(X)
Yhat_lab = np.argmax(Yhat, axis=-1)
loss, acc = model.evaluate(X, to_categorical(Y), batch_size=4, verbose=0)
print(loss)
print(acc)


# confusion matrix
cf=tf.math.confusion_matrix(Y.flatten(), Yhat_lab.flatten()).numpy() 
print(np.sum(np.diag(cf))/np.sum(cf)) # verify accuracy

print("base pair confusion (rows are ground truth)")
print(pd.DataFrame(cf, index=anc, columns=anc).to_latex())

# specificity (column-normalized, diagonal is fraction of A_hat which is truly A)
print("specificity (column-normalized) confusion")
print(pd.DataFrame(cf, index=anc, columns=anc).divide(cf.sum(axis=0), axis=1).round(decimals=3).to_latex())

# sensitivity (row-normalized, diagonal is fraction of A which we say is A_hat)
print("sensitivity (row-normalized) confusion")
print(pd.DataFrame(cf, index=anc, columns=anc).divide(cf.sum(axis=1), axis=0).round(decimals=3).to_latex())



# rinse and repeat with mode filter
print("---[MODE FILTER: WIDTH=4000]---")
mfw=2000
yhat2=np.zeros(Yhat_lab.shape)
for j in range(Yhat_lab.shape[-1]):
    yhat2[:,j]=ss.mode(Yhat_lab[:,max(0,j-mfw):min(Yhat_lab.shape[-1], j+mfw)], axis=1).mode.flatten()

print("accuracy")
cf2=tf.math.confusion_matrix(Y.flatten(), yhat2.flatten()).numpy()
print(np.sum(np.diag(cf2))/np.sum(cf2))

print("base pair confusion (rows are ground truth)")
print(pd.DataFrame(cf2, index=anc, columns=anc))

print("specificity (column-normalized) confusion")
print(pd.DataFrame(cf2, index=anc, columns=anc).divide(cf.sum(axis=0), axis=1).round(decimals=3).to_latex())

print("sensitivity (row-normalized) confusion")
print(pd.DataFrame(cf2, index=anc, columns=anc).divide(cf.sum(axis=1), axis=0).round(decimals=3).to_latex())
