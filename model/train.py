#!/usr/bin/env python
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
#import horovod.keras as hvd 
from segnet import segnet
from generator import DataGenerator,DataLoader


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
    tf.config.experimental.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # don't take up the whole gpu
    for d in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(d, True)
else: 
    # assume we're on sherlock -- load modules and check versions
    data_root='/scratch/users/magu/deepmix/data/'


# define functions
def load_train_set(chm=20, ix=0, count=int(1e9), bp1=0, bp2=int(1e9)):
    global data_root
    # subset variants
    V = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.V.npy')
    ix1 = max(ix, min(np.where(V[:,1].astype(int)-bp1 >= 0)[0]))
    ix2 = min(V.shape[0]+1, min(ix+count, max(np.where(V[:,1].astype(int)-bp2 <= 0)[0])))
    # load train data
    X = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.G.npy', mmap_mode='r')#[:,ix1:ix2,:]
    Y = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.L.npy', mmap_mode='r')#[:,ix1:ix2,:]
    S = np.load(data_root+'unzipped/panel_chr'+str(chm)+'.S.npy')
    # and indexes
    train=np.loadtxt('../data/reference-panel/split/train.strands.txt', dtype=str)
    train_ix=[i for i,q in enumerate(S) if q in train]
    np.random.shuffle(train_ix)
    print([X.shape, Y.shape, S.shape, len(train_ix)])
    return X, Y, S, V, train_ix, ix1, ix2



def load_dev_set(chm=20, ix=0, count=int(1e9), bp1=0, bp2=int(1e9)):
    global data_root
    # file paths
    x_f = data_root+'simulated_chr'+str(chm)+'/numpy/dev_10gen.no_OCE_WAS.query.npz'
    y_f = data_root+'simulated_chr'+str(chm)+'/label/dev_10gen.no_OCE_WAS.result.npz'
    # subset genetic data
    V = np.load(x_f)['V']
    ix1 = max(ix, min(np.where(V[:,1].astype(int)-bp1 >= 0)[0]))
    ix2 = min(V.shape[0], min(ix+count, max(np.where(V[:,1].astype(int)-bp2 <= 0)[0])))
    # load genetic data, then labels, making sure the sample ordering is the same
    sub=np.random.choice(np.arange(200), 120, replace=False) 
    S = np.load(data_root+'simulated_chr'+str(chm)+'/label/dev_10gen.no_OCE_WAS.result.npz')['S'][sub]
    S_f = np.load(x_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    X_dev = np.load(x_f)['G'][ids,:,:]#[:,ix1:ix2,:]
    S_f = np.load(y_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    # rfmix simulate has one-indexed labels, so the last slice is necessary
    Y_dev = to_categorical(np.load(y_f)['L'][ids,:], dtype='bool')[:,:,1:]#[:,ix1:ix2,:] 
    print([X_dev.shape, Y_dev.shape])
    return X_dev, Y_dev, S_f[ids]



def filter_ac(X, ac=1):
    # filters variants at >= ac in train set -- gives back indexes
    return (X.sum(axis=0) > ac).all(axis=1)
    

def train(chrom=20, out='segnet_weights', no_generator=False, batch_size=4, num_epochs=100,
          dropout_rate=0.01, input_dropout_rate=0.01, batch_norm=False, filter_size=8, 
          pool_size=4, num_blocks=5, num_filters=8, var_start=0, num_var=int(1e9), 
          bp_start=0, bp_end=int(1e9), array_only=False, continue_train=True, ivw=False,
          random_batch=False):
    ## Load data
    X, Y, S, V, train_ix, v1, v2 = load_train_set(chm=chrom, ix=var_start, count=num_var, bp1=bp_start, bp2=bp_end)
    X_dev, Y_dev, S_dev = load_dev_set(chm=chrom, ix=var_start, count=num_var, bp1=bp_start, bp2=bp_end)
    # filter variants, get counts of variants, alleles, ancestries
    vs=filter_ac(X[:,v1:v2,:], ac=2)
    if array_only: 
        # subset to MEGA variants
        x=np.loadtxt('../positions_on_mega_array.txt.gz', delimiter=' ', dtype=str)
        vs=vs & np.in1d(V[v1:v2,1], x[x[:,0]=='chr'+str(chrom),-1])
    nv = np.sum(vs) - (np.sum(vs) % (pool_size**num_blocks))
    na = X.shape[-1]
    vs = np.array([False for _ in range(v1-1)]+
                  [i and s <= nv for i,s in zip(vs,np.cumsum(vs))]+
                  [False for _ in range(v2,X.shape[1])]) # update truncation
    np.savetxt(out+'.var_index.txt', np.arange(len(vs))[vs], fmt='%i')
    # subset
    anc=np.array([0,1,2,3,5]) # ancestry indexes -- 4 is OCE, 6 is WAS
    X=X[np.ix_(train_ix, vs, np.arange(na))]
    Y=Y[np.ix_(train_ix, vs, anc)]
    X_dev=X_dev[:,vs,:na]
    Y_dev=Y_dev[np.ix_(np.arange(Y_dev.shape[0]), vs, np.arange(anc.shape[0]))]
    nc = Y.shape[-1]
     
    ## Create model, declare optimizer
    os.system('echo "pre-model"; nvidia-smi')
    #strategy = tf.distribute.experimental.CentralStorageStrategy()
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
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
   
    if continue_train and os.path.exists(out+'.h5') and os.path.exists(out+'.log.csv'):
        model.load_weights(out+'.h5')
        bb = np.genfromtxt(out+'.log.csv', delimiter=',')[-1,0] # subtract off previous batches
        print("continuing training from batch {}...".format(bb))
    else:
        bb = 0 # previous batches is zero

     
    ## Train model 
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    wt = callbacks.ModelCheckpoint(out+".h5", monitor='val_loss', mode='min', 
                                   save_freq='epoch', verbose=1, save_best_only=True)
    lg = callbacks.CSVLogger(out+'.log.csv', separator=",", append=continue_train)
    cw = Y.sum()/Y.sum(axis=0).sum(axis=0) if ivw else np.ones((Y.shape[-1],))
    if no_generator:
        history=model.fit(X, Y, validation_data=(X_dev, Y_dev), batch_size=batch_size, 
                          epochs=num_epochs - int(bb), callbacks=[es, wt, lg], class_weight=cw)
    else:
        bs=num_epochs - int(bb)
        params={'X':X, 'Y':Y, 'dim':nv, 'batch_size':bs, 'n_classes':nc, 'n_alleles':na}
        param2={'X':X_dev, 'Y':Y_dev, 'dim':nv, 'batch_size':bs, 'n_classes':nc, 'n_alleles':na}
        anc_fq=Y[:,0,:].sum(axis=0)
        anc_wt=((1/anc_fq)/((1/anc_fq).sum())).flatten() if random_train else np.ones((Y.shape[-1],))
        history=model.fit_generator(generator=DataGenerator(**params, sample=random_train, anc_wts=anc_wts), 
                                    validation_data=DataGenerator(**param2),
                                    epochs=num_epochs - int(bb), callbacks=[es, wt, lg], class_weight=cw)
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
    plt.savefig(out+'.info.png')


def get_args():
    import argparse
    parser=argparse.ArgumentParser(description=_README)
    parser.add_argument('--chrom', metavar='20', type=int, required=False,
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
    parser.add_argument('--array-only', action='store_true',
                         help='Flag to only use variants on the Illumina MEGA Array')
    parser.add_argument('--continue-train', action='store_true',
                         help='Flag to continue training from an existing model file')
    parser.add_argument('--ivw', action='store_true', 
                         help='Flag to weight classes by inverse frequency during training')
    parser.add_argument('--random-batch', action='store_true', 
                         help='Flag to take batch samples randomly (prop. to Y_label frequency)')
    #parser.add_argument('--n-alleles', metavar='na', type=int, nargs=1, required=False,
    #                     default=2,
    #                     help='Number of input alleles to consider')
    parser.add_argument('--num-var', metavar='nv', type=int, required=False, default=int(1e9),
                         help='Number of variants on chromosome to use (will be truncated to fit model specification)')
    parser.add_argument('--var-start', metavar='vs', type=int, required=False, default=0,
                         help='Index of variant to start training (i.e. data will be sliced from here to num-var beyond it)')
    parser.add_argument('--bp-start', metavar='bp1', type=int, required=False, default=0,
                         help='Starting base pair coordinate to subset input data')
    parser.add_argument('--bp-end', metavar='bp2', type=int, required=False, default=int(1e9),
                         help='Ending base pair coordinate to subset input data')
    #parser.add_argument('--multi-gpu', metavar='hor', action='store_true',
    #                     help='Flag to use horovod multi-gpu (also requires different script invocation)')
    parser.add_argument('--out', metavar='model_weights', type=str, required=True,
                         help='Output path prefix -- extensions automatically added')
    args=parser.parse_args()
    return args



def main():
    args=get_args()    
    # safety catch -- don't overwrite another model
    if os.path.exists(args.out+'.h5') and not args.continue_train:
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



