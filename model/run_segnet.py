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



# fair warning, all the paths in this script are broken (16 June 2020 -- MA)



## Load data
X = np.load('../unzipped/chm_21.genotypes.npy', mmap_mode='r')
Y = np.load('../unzipped/chm_21.labels.npy', mmap_mode='r')
S = np.load('../unzipped/chm_21.samples.npy')
print([X.shape, Y.shape, X.shape])


# get indexes of train set individuals
train=np.loadtxt('chm21.train.txt', dtype=str)
train_ix=[i for i,q in enumerate(train) if q in S]


# additional (random) dev set samples
n=100
d_ix=np.random.choice(np.arange(np.load('dev_10gen.result.npz').shape[0]), size=n, replace=False)
X_dev=np.load('dev_10gen.query.ALL_X.npz')['G'][d_ix,:n_variants,:n_variant_classes_per_variant]
# AMR is the first ancestry label, and there are none of them
Y_dev=to_categorical(np.load('dev_10gen.result.npz')['L'][d_ix,:n_variants], dtype='bool')[:,:,1:]
print([X_dev.shape, Y_dev.shape])
print("loaded data...")


## Additional variant filtration criteria (currently just subsetting)
n_variants = 2**18
n_variant_classes_per_variant = 3
n_classes = 4




## Create model, declare optimizer
model = segnet((n_variants, n_variant_classes_per_variant), n_classes)
adam_optimizer_fn = optimizers.Adam(lr=1e-4)


# hvd adjustments -- here and below
#adam_optimizer_fn = optimizers.Adam(lr=1e-4 * hvd.size())
#adam_optimizer_fn = hvd.DistributedOptimizer(adam_optimizer_fn)

# do a parallel thing (not hvd)
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
#with strategy.scope():
#    # Everything that creates variables should be under the strategy scope.
#    # In general this is only model construction & `compile()`.
#    model = segnet((n_variants, n_variant_classes_per_variant))
#    model.compile(optimizer=adam_optimizer_fn, loss='categorical_crossentropy', metrics=['accuracy']) 



## Compile model and summarize
model.compile(optimizer=adam_optimizer_fn, loss='categorical_crossentropy', metrics=['accuracy']) 

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
params = {'X': X, 'Y': Y, 
          'dim': n_variants,
          'batch_size': 128,
          'n_classes': n_classes,
          'n_variant_classes_per_variant': n_variant_classes_per_variant,
          'shuffle': True}

# (optional) do this with a generator object (generator.py)
#training_generator = DataGenerator(train_ix, **params)
#history = model.fit_generator(generator = training_generator, validation_data = (X_dev, Y_dev), use_multiprocessing=True, epochs=50)
#history = model.fit_generator(generator = training_generator, use_multiprocessing=True, epochs=50)


# do it
history=model.fit(X[train_ix,:n_variants,:n_variant_classes_per_variant], 
                  Y[train_ix,:n_variants,1:], batch_size=16, epochs=50, use_multiprocessing=True)



## Save model weights
model.save_weights("chm21all_saved_weights.h5")
print("saved weights!")




## Evaluate model -- not now tho
comment="""
# 1) create loss and accuracy plots
_, dev_acc = model.evaluate(X_dev, Y_dev, verbose=0)
# 1.1) plot loss during training
pyplot.figure(1, (9,9))
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# 1.2) plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig('acc_during_training.png')


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



