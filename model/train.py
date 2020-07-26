#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
#import horovod.keras as hvd 
from segnet import segnet
from generator import DataGenerator


_README="""
A script to train segnet model of local ancestry.

-Matthew Aguirre (magu[at]stanford[dot]edu)
"""


_TODO="""
1. Add trainset and devset paths to parser
2. Potentially number of variants and ancestry labels as well 
    - these are currently auto-inferred
3. Add early-stopping parameters to parser
4. And learning rate
5. Implement hvd multi-gpu??
"""


# quick check if we're on galangal
import platform
if platform.uname()[1]=='galangal.stanford.edu':
    data_root='/home/magu/deepmix/data/reference_panel/'
    # if so, use GPU #1
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    tf.config.experimental.set_visible_devices([], 'GPU')
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # don't take up the whole gpu
    # for d in tf.config.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(d, True)
else: 
    # assume we're on sherlock -- load modules and check versions
    data_root='/scratch/users/magu/deepmix/data/'


# define functions
def load_train_set(chm=20):
    global data_root
    # load train data
    X = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.G.npy', mmap_mode='r')
    Y = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.L.npy', mmap_mode='r')
    S = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.S.npy')
    # and indexes
    train=np.loadtxt('../data/reference-panel/split/train.strands.txt', dtype=str)
    train_ix=[i for i,q in enumerate(S) if q in train]
    np.random.shuffle(train_ix)
    print([X.shape, Y.shape, S.shape, len(train_ix)])
    return X, Y, S, train_ix



def load_dev_set(chm=20):
    global data_root
    # file paths
    x_f = data_root+'simulated_chr'+str(chm)+'/numpy/dev_10gen.query.npz'
    y_f = data_root+'simulated_chr'+str(chm)+'/label/dev_10gen.result.npz'
    # load genetic data, then labels, making sure the sample ordering is the same
    sub=np.random.choice(np.arange(200), 50, replace=False) 
    S = np.load(data_root+'simulated_chr'+str(chm)+'/label/dev_10gen.result.npz')['S'][sub]
    S_f = np.load(x_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    X_dev = np.load(x_f)['G'][ids,:,:]
    S_f = np.load(y_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    # rfmix simulate has one-indexed labels, so the last slice is necessary
    Y_dev = to_categorical(np.load(y_f)['L'][ids,:], dtype='bool')[:,:,1:] 
    print([X_dev.shape, Y_dev.shape])
    return X_dev, Y_dev, S_f[ids]



def train(chrom=20, out='segnet_weights', no_generator=False, batch_size=4, num_epochs=100,
          dropout_rate=0.01, input_dropout_rate=0.01, batch_norm=False, filter_size=8, 
          pool_size=4, num_blocks=5, num_filters=8):
    ## Load data
    X, Y, S, train_ix = load_train_set(chm=chrom)
    X_dev, Y_dev, S_dev = load_dev_set(chm=chrom)
    
    # get number of variants, alleles, and ancestries
    nv = X.shape[1] - (X.shape[1] % (pool_size**num_blocks)) # truncation by up to 1024 
    na = X.shape[-1]
    nc = Y.shape[-1]
    
    ## Create model, declare optimizer
    os.system('echo "pre-model"; nvidia-smi')
    model = segnet(input_shape=(nv,na), n_classes=nc, 
                   width=filter_size, n_filters=num_filters, pool_size=pool_size, 
                   n_blocks=num_blocks, dropout_rate=dropout_rate, 
                   input_dropout_rate=input_dropout_rate, l2_lambda=1e-30, 
                   batch_normalization=batch_norm)
    adam = optimizers.Adam(lr=1e-4)
    os.system('echo "post-compile"; nvidia-smi')

    ## Compile model and summarize
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
    print(model.summary())    
    
    ## Train model 
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    if no_generator:
        history=model.fit(X[train_ix,:nv,:na], Y[train_ix,:nv,:], 
                          validation_data=(X_dev[:,:nv,:], Y_dev[:,:nv,:]),
                          batch_size=batch_size, epochs=num_epochs, callbacks=[es])
    else:
        params={'X':X, 'Y':Y, 'dim':nv, 'batch_size':batch_size, 'n_classes':nc, 'n_alleles':na}
        param2={'X':X_dev, 'Y':Y_dev, 'dim':nv, 'batch_size':batch_size, 'n_classes':nc, 'n_alleles':na}
        train_gen=DataGenerator(train_ix, **params)
        valid_gen=DataGenerator(np.arange(X_dev.shape[0]), **param2)
        history=model.fit_generator(generator=train_gen, validation_data=valid_gen, 
                                    epochs=num_epochs, callbacks=[es])
    ## Save model weights and return
    model.save_weights(out+'.h5')
    return history

    

def plot_info(history, out):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # plot loss during training
    plt.figure(1, (9,9))
    plt.subplot(211)
    plt.title('Loss during training')
    plt.plot(history.history['loss'], label='train set')
    plt.plot(history.history['val_loss'], label='dev set')
    plt.legend()

    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train set')
    plt.plot(history.history['val_accuracy'], label='dev set')
    plt.legend()
    plt.savefig(out+'info.png')



def main():
    import argparse
    parser=argparse.ArgumentParser(description=_README)
    parser.add_argument('--chrom', metavar='20', type=int, nargs=1, required=False,
                         default=20,
                         help='Chromosome to use (must be in 1,2,...,22)')
    parser.add_argument('--batch-size', metavar='4', type=int, required=False,
                         default=4,
                         help='Minibatch size for training')
    parser.add_argument('--num-filters', metavar='8', type=int, required=False,
                         default=8,
                         help='Number of filters in first segnet layer')
    parser.add_argument('--filter-size', metavar='16', type=int, required=False,
                         default=16,
                         help='Convolutional filter size in segnet')
    parser.add_argument('--num-epochs', metavar='100', type=int, required=False,
                         default=100,
                         help='Number of epochs to train model')
    parser.add_argument('--num-blocks', metavar='5', type=int, required=False,
                         default=5,
                         help='Number of down/upward blocks (equivalent to model depth)')
    parser.add_argument('--pool-size', metavar='4', type=int, required=False,
                         default=4,
                         help='Width of maxpool operator')
    parser.add_argument('--dropout-rate', metavar='0.01', type=float, required=False,
                         default=0.01,
                         help='Dropout rate at each layer')
    parser.add_argument('--input-dropout-rate', metavar='0.01', type=float, required=False,
                         default=0.01,
                         help='Dropout rate after input layer')
    parser.add_argument('--batch-norm', action='store_true',
                         help='Flag to use batch normalization')
    parser.add_argument('--no-generator', action='store_true',
                         help='Flag to not use generator object, and load all data into memory')
    #parser.add_argument('--n-alleles', metavar='na', type=int, nargs=1, required=False,
    #                     default=2,
    #                     help='Number of input alleles to consider')
    #parser.add_argument('--nvar', metavar='nv', type=int, nargs=1, required=True,
    #                     help='Number of variants on chromosome to use (must be power of 2)')
    #parser.add_argument('--multi-gpu', metavar='hor', action='store_true',
    #                     help='Flag to use horovod multi-gpu (also requires different script invocation)')
    parser.add_argument('--out', metavar='model_weights', type=str, required=True,
                         help='Output path prefix -- extensions automatically added')
    args=parser.parse_args()
    
    # safety catch -- don't overwrite another model
    if os.path.exists(args.out+'.h5'):
        print(args.out+'.h5 object already found. Aborting!')
        exit(1)
    
    # train model, plot info
    print(args)
    history=train(**vars(args))
    plot_info(history, args.out)
    return
    


if __name__=='__main__':
    main()





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



