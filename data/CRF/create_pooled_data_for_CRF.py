#!/usr/bin/env python
import numpy as np
import sklearn_crfsuite
import random
from scipy import stats
import math
import pickle



_README="""
A script to generate pooled data for chain CRF. 
-Jan Sokol (jsokol[at]stanford[dot]edu)
"""


_TODO="""
"""


# quick check if we're on galangal
import platform
if platform.uname()[1]=='galangal.stanford.edu':
    print('Error: script written to be run on sherlock')
else: 
    # assume we're on sherlock
    print('Assuming we are on sherlock')

    

# define functions
def load_input_data(args): 
    X_train = np.load(vars(args)['Segnet_pred_trainset']) 
    X_dev = np.load(vars(args)['Segnet_pred_devset']) 
    Y_train = np.load(vars(args)['trainset_labels']) 
    Y_dev = np.load(vars(args)['devset_labels']) 
    return X_train, Y_train, X_dev, Y_dev



def average_windows(X_train, Y_train, X_dev, Y_dev, args):
    X_train_new=np.zeros((X_train.shape[0], math.floor(X_train.shape[1]/vars(args)['pooling_window']), X_train.shape[2]))
    Y_train_new=np.zeros((Y_train.shape[0], math.floor(Y_train.shape[1]/vars(args)['pooling_window']), Y_train.shape[2]))
    X_dev_new=np.zeros((X_dev.shape[0], math.floor(X_dev.shape[1]/vars(args)['pooling_window']), X_dev.shape[2]))
    Y_dev_new=np.zeros((Y_dev.shape[0], math.floor(Y_dev.shape[1]/vars(args)['pooling_window']), Y_dev.shape[2]))
    for train_vs_dev_i in [1,0]:
        for individual_i in range([X_train, X_dev][train_vs_dev_i].shape[0]):
            if (individual_i+1)%100==0 or individual_i+1==1:
                print('Pooling windows for sample '+str(individual_i+1)+' of '+str([X_train, X_dev][train_vs_dev_i].shape[0]))
            for window_i in np.arange(0+(vars(args)['pooling_window']-1), [X_train, X_dev][train_vs_dev_i].shape[1], vars(args)['pooling_window']):
                for label_i in range([X_train, X_dev][train_vs_dev_i].shape[2]):
                    [X_train_new, X_dev_new][train_vs_dev_i][individual_i,int((window_i+1)/vars(args)['pooling_window'])-1,label_i] = np.mean([X_train, X_dev][train_vs_dev_i][individual_i,window_i-(vars(args)['pooling_window']-1):window_i+1,label_i])
                [Y_train_new, Y_dev_new][train_vs_dev_i][individual_i,int((window_i+1)/vars(args)['pooling_window'])-1,stats.mode(np.argmax([Y_train, Y_dev][train_vs_dev_i][individual_i,window_i-(vars(args)['pooling_window']-1):window_i+1,:],axis=-1))[0][0]]=1
    Y_train_nonsimp, Y_dev_nonsimp = Y_train, Y_dev
    X_train, Y_train, X_dev, Y_dev = X_train_new, Y_train_new, X_dev_new, Y_dev_new
    X_train_new, Y_train_new, X_dev_new, Y_dev_new = [], [], [], []
    print([X_train.shape, Y_train.shape, Y_train_nonsimp.shape, X_dev.shape, Y_dev.shape, Y_dev_nonsimp.shape])
    return X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp
    
    

def evaluate_raw_model_accuracy(X_dev, Y_dev):
    X_dev_indexed, Y_dev_indexed = np.argmax(X_dev, axis=-1), np.argmax(Y_dev, axis=-1)
    counter, missclassification_counter = 0, 0
    for individual_i in range(X_dev_indexed.shape[0]):
        for base_i in range(X_dev_indexed.shape[1]):
            if X_dev_indexed[individual_i, base_i]==Y_dev_indexed[individual_i, base_i]:
                counter += 1
            else:
                counter += 1
                missclassification_counter += 1
    X_dev_indexed, Y_dev_indexed = [], []
    print('raw model accuracy before format conversion: '+ str((counter-missclassification_counter)/counter))

    
    
def load_recprobs(X_train, args): 
    original_rec_probs = np.genfromtxt('/scratch/users/jsokol/deepLAI/CRF/hapmap-phase2-genetic-map.tsv', dtype=str)
    original_rec_probs = original_rec_probs[original_rec_probs[:,0]=='20'][:,1:].astype('float')
    v_rec_probs = np.load('/scratch/users/magu/deepmix/data/simulated_chr20/numpy/train_10gen.no-OCE-WAS.big.npz')['V']
    v_rec_probs = v_rec_probs[:X_train.shape[1]*vars(args)['pooling_window']+1,1][:,np.newaxis]
    v_rec_probs = np.pad(v_rec_probs, ((0,0),(0,1)), mode='constant', constant_values=0).astype('float')
    # now for each variant position find the previous and next rec probability and interpolate
    upperequal_bound_i=0
    for variant_pos_i in range(v_rec_probs.shape[0]):
        while int(original_rec_probs[upperequal_bound_i,0]) < int(v_rec_probs[variant_pos_i,0]):
            upperequal_bound_i += 1
        if int(original_rec_probs[upperequal_bound_i,0])==int(v_rec_probs[variant_pos_i,0]):
            v_rec_probs[variant_pos_i,1] = original_rec_probs[upperequal_bound_i,1]
        else:
            # interpolate
            v_rec_probs[variant_pos_i,1] = ((original_rec_probs[upperequal_bound_i,0]-v_rec_probs[variant_pos_i,0])*original_rec_probs[upperequal_bound_i-1,1]
        + (v_rec_probs[variant_pos_i,0]-original_rec_probs[upperequal_bound_i-1,0])*original_rec_probs[upperequal_bound_i,1])/(original_rec_probs[upperequal_bound_i,0]-original_rec_probs[upperequal_bound_i-1,0])
    v_rec_probs = v_rec_probs[:,1]
    print(v_rec_probs.shape)
    return v_rec_probs
        
    
    
def create_CRF_data(X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp, v_rec_probs, args):
    # create data for linear-chain CRF
    norecprob=vars(args)['use_recprob']==False
    n_variants=int(X_train.shape[1]-1)
    X_train_new, Y_train_new, Y_train_nonsimp_new, X_dev_new, Y_dev_new, Y_dev_nonsimp_new = list(), list(), list(), list(), list(), list()
    for set_i in [1,0]:
        for individual_i in range([X_train.shape[0], X_dev.shape[0]][set_i]):
            if (individual_i+1)%100==0 or individual_i+1==1:
                print('Converting to correct format for sample '+str(individual_i+1)+' of '+str([X_train.shape[0], X_dev.shape[0]][set_i]))
            example=list()
            example_labels=list()
            for base_i in range(2,n_variants-2): 
                datapoint=dict()
                for anc_prob_i in range(0,X_train.shape[2]):
                    datapoint[str(anc_prob_i)]=[X_train, X_dev][set_i][individual_i,base_i,anc_prob_i]
                    if vars(args)['deep']:
                        datapoint[str(anc_prob_i)+'-1']=[X_train, X_dev][set_i][individual_i,base_i-1,anc_prob_i]
                        datapoint[str(anc_prob_i)+'-2']=[X_train, X_dev][set_i][individual_i,base_i-2,anc_prob_i]
                        datapoint[str(anc_prob_i)+'+1']=[X_train, X_dev][set_i][individual_i,base_i+1,anc_prob_i]
                        datapoint[str(anc_prob_i)+'+2']=[X_train, X_dev][set_i][individual_i,base_i+2,anc_prob_i]
                if norecprob==False:
                    datapoint[str(5)] = v_rec_probs[(base_i+1)*vars(args)['pooling_window']-1-math.floor(vars(args)['pooling_window']/2-0.5)] - v_rec_probs[(base_i-1)*vars(args)['pooling_window']+math.floor(vars(args)['pooling_window']/2-0.5)]
                if vars(args)['deep']:
                    datapoint[str(5)+'-1'] = v_rec_probs[(base_i+1-1)*vars(args)['pooling_window']-1-math.floor(vars(args)['pooling_window']/2-0.5)] - v_rec_probs[(base_i-1-1)*vars(args)['pooling_window']+math.floor(vars(args)['pooling_window']/2-0.5)]
                    datapoint[str(5)+'-2'] = v_rec_probs[(base_i+1-2)*vars(args)['pooling_window']-1-math.floor(vars(args)['pooling_window']/2-0.5)] - v_rec_probs[(base_i-1-2)*vars(args)['pooling_window']+math.floor(vars(args)['pooling_window']/2-0.5)]
                    datapoint[str(5)+'+1'] = v_rec_probs[(base_i+1+1)*vars(args)['pooling_window']-1-math.floor(vars(args)['pooling_window']/2-0.5)] - v_rec_probs[(base_i-1+1)*vars(args)['pooling_window']+math.floor(vars(args)['pooling_window']/2-0.5)]
                    datapoint[str(5)+'+2'] = v_rec_probs[(base_i+1+2)*vars(args)['pooling_window']-1-math.floor(vars(args)['pooling_window']/2-0.5)] - v_rec_probs[(base_i-1+2)*vars(args)['pooling_window']+math.floor(vars(args)['pooling_window']/2-0.5)]
                example.append(datapoint)
                example_labels.append(str(np.argmax([Y_train, Y_dev][set_i][individual_i, base_i, :])))
            
            example_labels_nonsimp=list()
            for base_i in range([Y_train_nonsimp, Y_dev_nonsimp][set_i].shape[1]):
                example_labels_nonsimp.append(str(np.argmax([Y_train_nonsimp, Y_dev_nonsimp][set_i][individual_i, base_i, :])))
            
            [X_train_new, X_dev_new][set_i].append(example)
            [Y_train_new, Y_dev_new][set_i].append(example_labels)
            [Y_train_nonsimp_new, Y_dev_nonsimp_new][set_i].append(example_labels_nonsimp)
                
    original_rec_probs, v_rec_probs = [], []
    X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp = X_train_new, Y_train_new, Y_train_nonsimp_new, X_dev_new, Y_dev_new, Y_dev_nonsimp_new
    X_train_new, Y_train_new, Y_train_nonsimp_new, X_dev_new, Y_dev_new, Y_dev_nonsimp_new = [], [], [], [], [], []
    print([len(X_train), len(Y_train), len(Y_train_nonsimp), len(X_dev), len(Y_dev), len(Y_dev_nonsimp)])
    return X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp
    
    
    
def get_args():
    import argparse
    parser=argparse.ArgumentParser(description=_README)
    parser.add_argument('--pooling_window', metavar='1000', type=int, required=False, default=1000,
                         help='pooling window')
    parser.add_argument('--use_recprob', action='store_true',
                         help="use recombination probabilities (doesn't aid accuracy)")
    parser.add_argument('--deep', action='store_true',
                         help="store class probabilities of examples more than one position behind (doesn't aid accuracy)")
    parser.add_argument('--outpath', metavar='output path prefix', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/chain_CRF/data/', help="store class probabilities of examples more than one position behind (doesn't aid accuracy)")
    parser.add_argument('--Segnet_pred_trainset', metavar='path to SegNet predictions over trainset (.npy file)', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/raw_model_outputs/Y_hat_segnet_train_10gen.no_OCE_WAS.npy', help='path to SegNet predictions over trainset (.npy file)')
    parser.add_argument('--Segnet_pred_devset', metavar='path to SegNet predictions over devset (.npy file)', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/raw_model_outputs/Y_hat_segnet_dev_10gen.no_OCE_WAS.npy', help='path to SegNet predictions over devset (.npy file)')
    parser.add_argument('--trainset_labels', metavar='path to trainset labels (.npy file)', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/raw_model_outputs/Y_train_10gen.no_OCE_WAS.npy', help='path to trainset labels (.npy file)')
    parser.add_argument('--devset_labels', metavar='path to devset labels (.npy file)', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/raw_model_outputs/Y_dev_10gen.no_OCE_WAS.npy', help='path to devset labels (.npy file)')  
    args=parser.parse_args()
    return args



def main():
    args=get_args()
    print(args)
    X_train, Y_train, X_dev, Y_dev = load_input_data(args)              
    X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp = average_windows(X_train, Y_train, X_dev, Y_dev, args)
    evaluate_raw_model_accuracy(X_dev, Y_dev)
    if vars(args)['use_recprob']:
        v_rec_probs = load_recprobs(X_train, args)    
    else:
        v_rec_probs = None
    X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp = create_CRF_data(X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp, v_rec_probs, args)
    # evaluate raw SegNet's accuracy as sanity check
    counter, missclassification_counter = 0, 0
    for example_i in range(len(X_dev)):
        for base_i in range(len(X_dev[example_i])):
            if str(max(['0','1','2','3','4'], key=X_dev[example_i][base_i].get))==str(Y_dev[example_i]):
                counter += 1
            else:
                counter += 1
                missclassification_counter += 1
    print('Evaluated raw model accuracy (sanity check): '+ str((counter-missclassification_counter)/counter))
    # shuffle data 
    train_shuffle_list, dev_shuffle_list = list(zip(X_train, Y_train, Y_train_nonsimp)), list(zip(X_dev, Y_dev, Y_dev_nonsimp))
    random.shuffle(train_shuffle_list), random.shuffle(dev_shuffle_list)
    (X_train, Y_train, Y_train_nonsimp), (X_dev, Y_dev, Y_dev_nonsimp) = zip(*train_shuffle_list), zip(*dev_shuffle_list)
    # store data
    if vars(args)['use_recprob']==False:
        norecprob_str='norecprob_'
    else:
        norecprob_str=''
    if vars(args)['deep']:
        deep_str='deep_'
    else:
        deep_str=''
    for i in range(len(X_train)):
        if (1+i)%100==0 or 1+i==1:
            print('Storing '+str(i+1)+'th sample of trainset')
        pickle.dump(X_train[i], open(vars(args)['outpath']+'X_train_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
        pickle.dump(Y_train[i], open(vars(args)['outpath']+'Y_train_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
        pickle.dump(Y_train_nonsimp[i], open(vars(args)['outpath']+'Y_train_nonsimp_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
    for i in range(len(X_dev)):
        if (i+1)%100==0 or 1+i==1:
            print('Storing '+str(i+1)+'th sample of devset')
        pickle.dump(X_dev[i], open(vars(args)['outpath']+'X_dev_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
        pickle.dump(Y_dev[i], open(vars(args)['outpath']+'Y_dev_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
        pickle.dump(Y_dev_nonsimp[i], open(vars(args)['outpath']+'Y_dev_nonsimp_'+str(vars(args)['pooling_window'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle','wb'))
    


if __name__=='__main__':
    main()




