import numpy as np
import keras
import scipy.stats as ss
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_ix, batch_size, dim, n_alleles, n_classes, X, Y, 
                 shuffle=True, admix=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.array(train_ix, dtype=int)
        if shuffle:
            np.random.shuffle(self.list_IDs)
        self.n_alleles = n_alleles
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.X = X
        self.Y = Y
        self.admix = admix
        self.indexes = np.arange(self.list_IDs.shape[0])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        # Find list of IDs
        list_IDs_temp = np.array([self.list_IDs[k] for k in indexes])
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        X_sm = self.X[list_IDs_temp, :self.dim, :self.n_alleles] 
        y_sm = self.Y[list_IDs_temp, :self.dim, :]         
        if self.admix:
            return naive_admixing(X_sm, y_sm)
        else:
            return X_sm, y_sm


def naive_admixing(X_data, Y_data):
    'Takes X_data and Y_data (labels) and returns naively admixed samples'
    n_fake=X_data.shape[0]
    maxgen=3
    n_splits=2+np.hstack([ss.poisson.rvs(2.86*gen, size=n_fake//maxgen) for gen in range(1,maxgen)])

    # new individuals
    new_X=[]
    new_Y=[]
    for j in n_splits:
        if j==0:
            ind=np.random.choice(np.arange(X_data.shape[0]), size=1)
            new_X.append(list(X_data[ind,:,:]))
            new_Y.append(list(Y_data[ind,:]))
        # sample breakpoints uniformly
        breaks=np.sort(X_data.shape[1] * ss.beta.rvs(a=1, b=1, size=j)).astype(int)
        # pick founders uniformly at random and stitch their labels together
        founds=np.random.choice(np.arange(X_data.shape[0]), size=j+1, replace=True)
        # assemble genome and labels
        new_x,new_y = [],[]
        new_x.append(X_data[founds[0],:breaks[0],:])
        new_y.append(Y_data[founds[0],:breaks[0]])
        for i,found in enumerate(founds[1:-1]):
            new_x.append(X_data[found, breaks[i]:breaks[i+1],:])
            new_y.append(Y_data[found, breaks[i]:breaks[i+1]])
        new_x.append(X_data[founds[-1], breaks[-1]:,:])
        new_y.append(Y_data[founds[-1], breaks[-1]:])
        new_X.append(np.vstack(new_x))
        new_Y.append(np.vstack(new_y))
    X_data=np.vstack((X_data, new_X))
    Y_data=np.vstack((Y_data, new_Y))
    
    return X_data, Y_data

    
    
    
