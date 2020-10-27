#!/usr/bin/env python
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from segnet import segnet
from generator import DataGenerator,DataLoader


_README="""
A script to train segnet model of local ancestry.

-Matthew Aguirre (magu[at]stanford[dot]edu)
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
    train=np.loadtxt('../data/reference-panel/split/train.strands.no-oce-was.txt', dtype=str)
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
    sub=np.random.choice(np.arange(200), 200, replace=False) 
    S = np.load(data_root+'simulated_chr'+str(chm)+'/label/dev_10gen.no_OCE_WAS.result.npz')['S'][sub]
    S_f = np.load(x_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    X_dev = np.load(x_f)['G'][ids,:,:]#[:,ix1:ix2,:]
    S_f = np.load(y_f)['S']
    ids = [np.where(S_f==(i))[0][0] for i in S]
    # rfmix simulate has one-indexed labels, so the last slice is necessary
    #Y_tmp = np.load(y_f, mmap_mode='r')['L'][ids,:]
    #Y_dev = np.dstack([Y_tmp==i for i in range(1,Y_tmp.shape[-1]+1)])
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
          random_batch=False, admix=False):
    ## Load data
    X, Y, S, V, train_ix, v1, v2 = load_train_set(chm=chrom, ix=var_start, count=num_var, bp1=bp_start, bp2=bp_end)
    X_dev, Y_dev, S_dev = load_dev_set(chm=chrom, ix=var_start, count=num_var, bp1=bp_start, bp2=bp_end)
    # filter variants, get counts of variants, alleles, ancestries
    vs=filter_ac(X[:,v1:v2,:], ac=1)
    nv = np.sum(vs) - (np.sum(vs) % (pool_size**num_blocks))
    na = X.shape[-1]
    vs = np.array([False for _ in range(v1-1)]+
                  [i and s <= nv for i,s in zip(vs,np.cumsum(vs))]+
                  [False for _ in range(v2,X.shape[1])]) # update truncation
    if os.path.exists(out+'.var_index.txt'):
        vs=np.genfromtxt(out+'.var_index.txt', dtype=int)
        nv=vs.shape[0]
    else:
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
                                   verbose=1, save_best_only=True)
    lg = callbacks.CSVLogger(out+'.log.csv', separator=",", append=continue_train)
    cw = np.sqrt(1/Y.sum(axis=0).sum(axis=0)) if ivw else np.ones((Y.shape[-1],))
    if no_generator:
        history=model.fit(X, Y, validation_data=(X_dev, Y_dev), batch_size=batch_size, 
                          epochs=num_epochs - int(bb), callbacks=[es, wt, lg], class_weight=cw)
    else:
        params={'X':X, 'Y':Y, 'dim':nv, 'batch_size':batch_size, 'n_classes':nc, 'n_alleles':anc.shape[0], 
                'train_ix':np.arange(X.shape[0])}
        param2={'X':X_dev, 'Y':Y_dev, 'dim':nv, 'batch_size':batch_size, 'n_classes':nc, 'n_alleles':anc.shape[0],
                'train_ix':np.arange(X_dev.shape[0])}
        anc_fq=Y[:,0,:].sum(axis=0)
        anc_wt=((1/anc_fq)/((1/anc_fq).sum())).flatten() if random_batch else np.ones((Y.shape[-1],))
        history=model.fit_generator(generator=DataGenerator(**params, sample=random_batch, anc_wts=anc_wt, admix=admix), 
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
    parser.add_argument('--continue-train', action='store_true',
                         help='Flag to continue training from an existing model file')
    parser.add_argument('--ivw', action='store_true', 
                         help='Flag to weight classes by inverse frequency during training')
    parser.add_argument('--admix', action='store_true', help='Flag to use admixed individuals during training')
    parser.add_argument('--random-batch', action='store_true', 
                         help='Flag to take batch samples randomly (prop. to Y_label frequency)')
    parser.add_argument('--num-var', metavar='nv', type=int, required=False, default=int(1e9),
                         help='Number of variants on chromosome to use (will be truncated to fit model specification)')
    parser.add_argument('--var-start', metavar='vs', type=int, required=False, default=0,
                         help='Index of variant to start training (i.e. data will be sliced from here to num-var beyond it)')
    parser.add_argument('--bp-start', metavar='bp1', type=int, required=False, default=0,
                         help='Starting base pair coordinate to subset input data')
    parser.add_argument('--bp-end', metavar='bp2', type=int, required=False, default=int(1e9),
                         help='Ending base pair coordinate to subset input data')
    parser.add_argument('--out', metavar='model_weights', type=str, required=True,
                         help='Output path prefix -- extensions automatically added')
    args=parser.parse_args()
    return args


## define main method, and run if applicable
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





