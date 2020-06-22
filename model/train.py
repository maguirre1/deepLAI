#!/usr/bin/env python
import os
import numpy as np
import scipy.stats as ss
import tensorflow as tf
from keras import optimizers, callbacks, regularizers
from keras.models import Model
from keras.utils import to_categorical
#import horovod.keras as hvd 
from segnet import segnet
from generator import DataGenerator



## Set variant filtration criteria (just subsetting)
nv = int(2**20) # variants
na = 3          # alleles
nc = 4          # ancestries 
bs = 4          # batch size
ge = True       # use generator object
nf = 8          # number of filters for segnet
ne = 100        # number of epochs
# todo: give this a command-line interface


## Load data
data_root='/home/magu/deepmix/data/ALL_DNA_dataset/'
X = np.load(data_root+'unzipped/chm_21.genotypes.npy', mmap_mode='r')
Y = np.load(data_root+'unzipped/chm_21.labels.npy', mmap_mode='r')
S = np.load(data_root+'unzipped/chm_21.samples.npy')
print([X.shape, Y.shape, X.shape])


# get indexes of train set individuals
train=np.loadtxt(data_root+'chm21.train.txt', dtype=str)
train_ix=[i for i,q in enumerate(train) if q in S]


# additional (random) dev set samples -- first choose indexes
n=100
S=np.load(data_root+'simulated/label/dev_10gen.result.npz')['S']
s=np.random.choice(S, size=n, replace=False)

# then load and subset -- AMR is the first ancestry label, ignored for now
x_f=data_root+'simulated/numpy/dev_10gen.query.ALL_X.npz'
y_f=data_root+'simulated/label/dev_10gen.result.npz'
S_f=np.load(x_f)['S']
X_dev=np.load(x_f)['G'][[np.where(S_f==(i))[0][0] for i in s],:nv,:na]
S_f=np.load(y_f)['S']
Y_dev=to_categorical(np.load(y_f)['L'][[np.where(S_f==(i))[0][0] for i in s],:nv], dtype='bool')[:,:,1:]
print([X_dev.shape, Y_dev.shape])
print("loaded data...")



## Create model, declare optimizer
model = segnet(input_shape=(nv,na), n_classes=nc, n_filters=nf)
adam = optimizers.Adam(lr=1e-4)


# hvd adjustments -- here and below
#adam = optimizers.Adam(lr=1e-4 * hvd.size())
#adam = hvd.DistributedOptimizer(adam)

# do a parallel thing (not hvd)
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
#with strategy.scope():
#    # Everything that creates variables should be under the strategy scope.
#    # In general this is only model construction & `compile()`.
#    model = segnet((n_variants, n_alleles))
#    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 



## Compile model and summarize
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
print(model.summary())
# print('Learning rate = ' + str(K.eval(model.optimizer.lr)))


# more multi-gpu stuff
no_call="""
# Horovod: broadcast initial states from rank 0 to all other processes 
callback = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
   callback.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
"""



## Train model 
print("training model...")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
if ge: 
    params={'X':X, 'Y':Y, 'dim':nv, 'batch_size':bs, 'n_classes':nc, 'n_alleles':na}
    generator=DataGenerator(train_ix, **params)
    history=model.fit_generator(generator=generator, validation_data=(X_dev, Y_dev), 
                                epochs=ne, callbacks=[es])
else:
    history=model.fit(X[train_ix,:nv,:na], Y[train_ix,:nv,1:], validation_data=(X_dev, Y_dev),
                      batch_size=bs, epochs=ne, callbacks=[es])



## Save model weights
model.save_weights("chm21all_saved_weights.h5")
print("saved weights!")



## Plot loss during training -- with final devset accuracy
_, dev_acc = model.evaluate(X_dev, Y_dev, verbose=0)

# 1.1) plot loss during training
import matplotlib.pyplot as plt
plt.figure(1, (9,9))
plt.subplot(211)
plt.title('Loss during training')
plt.plot(history.history['loss'], label='train set')
plt.plot(history.history['val_loss'], label='dev set')
plt.legend()

# 1.2) plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train set')
plt.plot(history.history['val_accuracy'], label='dev set')
plt.legend()
plt.savefig('acc_during_training.png')



## Evaluate model -- not now tho
comment="""
# print out confusion matrix of predicted classifications
y_pred_dev = model.predict(X_dev, verbose=1)
y_pred_dev_flattened = np.argmax(y_pred_test, axis=-1).flatten()
Y_dev_groundtruths_flattened = np.argmax(Y_dev, axis=-1).flatten()
output_confusion_matrix = confusion_matrix(Y_dev_groundtruths_flattened, y_pred_dev_flattened)
y_pred_dev_flattened = Y_dev_groundtruths_flattened = []
print(output_confusion_matrix)


# In[ ]:


# display ground truth admixtures and predicted admixtures in a small subset of dev set
pyplot.figure(figsize=(12, 8))
Y_dev_groundtruths = np.argmax(Y_data[1], axis=-1)
pyplot.subplot(211)
pyplot.title('Small subsample of dev set ground truths')
pyplot.imshow(Y_dev_groundtruths[350:365,:].astype(int), aspect='auto', cmap='jet')
pyplot.savefig('sample_dev_trues.png')

y_pred_dev = np.argmax(y_pred_dev, axis=-1)
pyplot.subplot(212)
pyplot.title('Corresponding dev set predictions')
pyplot.imshow(y_pred_dev[350:365,:].astype(int), aspect='auto', cmap='jet')
pyplot.savefig('sample_dev_preds.png')

# In[ ]:


# store outputs 
from numpy import save
save('Y_dev_chm21all.npy', Y_dev)
save('X_dev_chm21all.npy', X_dev)


# In[ ]:


# load model and other vital files to re-run this script
# model.load_weights("stored_models/chm21all_saved_weights.h5")
# Y_dev = np.load('stored_models/Y_dev_chm21all.npy', Y_dev)
# X_dev = np.load('stored_models/X_dev_chm21all.npy', X_dev)

# In[ ]:
"""



