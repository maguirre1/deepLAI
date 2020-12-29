#!/usr/bin/env python
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import random
import pickle
from os import path



_README="""
A script to train a linear-chain CRF and evaluate its accuracy. 
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
def load_data(args):
    print('Loading data')
    trainset_example_n=3000
    devset_example_n=200
    data_path=vars(args)['input_path']
    X_train, Y_train, X_dev, Y_dev = list(), list(), list(), list()
    if vars(args)['win']==1:
        for i in range(trainset_example_n):
            if (i+1)%100==0 or i+1==1:
                print('Loading '+str(i+1)+'th example of trainset')
            if vars(args)['recprob']==False:
                X_train.append(pickle.load(open(data_path+'X_train_short_norecprob_'+str(i)+'.pickle',"rb")))
                Y_train.append(pickle.load(open(data_path+'Y_train_short_norecprob_'+str(i)+'.pickle',"rb")))
            else:
                X_train.append(pickle.load(open(data_path+'X_train_short_'+str(i)+'.pickle',"rb")))
                Y_train.append(pickle.load(open(data_path+'Y_train_short_'+str(i)+'.pickle',"rb")))
        for i in range(devset_example_n):
            if (i+1)%100==0 or i+1==1:
                print('Loading '+str(i+1)+'th example of devset')
            if vars(args)['recprob']==False:
                X_dev.append(pickle.load(open(data_path+'X_dev_short_norecprob_'+str(i)+'.pickle',"rb")))
                Y_dev.append(pickle.load(open(data_path+'Y_dev_short_norecprob_'+str(i)+'.pickle',"rb")))
            else:
                X_dev.append(pickle.load(open(data_path+'X_dev_short_'+str(i)+'.pickle',"rb")))
                Y_dev.append(pickle.load(open(data_path+'Y_dev_short_'+str(i)+'.pickle',"rb")))
    else:
        Y_train_nonsimp, Y_dev_nonsimp = list(), list()
        if vars(args)['recprob']==False:
            norecprob_str='norecprob_'
        else:
            norecprob_str=''
        if vars(args)['deep']:
            deep_str='deep_'
        else:
            deep_str=''
        for i in range(trainset_example_n):
            if (i+1)%100==0 or i+1==1:
                print('Loading '+str(i+1)+'th example of trainset')
            X_train.append(pickle.load(open(data_path+'X_train_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
            Y_train.append(pickle.load(open(data_path+'Y_train_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
            Y_train_nonsimp.append(pickle.load(open(data_path+'Y_train_'+'nonsimp_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
        for i in range(devset_example_n):
            if (i+1)%100==0 or i+1==1:
                print('Loading '+str(i+1)+'th example of devset')
            X_dev.append(pickle.load(open(data_path+'X_dev_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
            Y_dev.append(pickle.load(open(data_path+'Y_dev_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
            Y_dev_nonsimp.append(pickle.load(open(data_path+'Y_dev_'+'nonsimp_'+str(vars(args)['win'])+'win_'+norecprob_str+deep_str+str(i)+'.pickle',"rb")))
    return X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp
    
    

def define_model(args):
    if vars(args)['alg']=='lbfgs':
        c1=vars(args)['c1']
        c2=vars(args)['c2']
    elif vars(args)['alg']=='l2sgd':
        c1=None
        c2=vars(args)['c2']
    else:
        c1=None
        c2=None
    model = sklearn_crfsuite.CRF(
    algorithm=vars(args)['alg'], # try 'l2sgd'
    c1=c1,
    c2=c2,
    max_iterations=vars(args)['it'], 
    all_possible_transitions=vars(args)['all_trans'],
    all_possible_states=vars(args)['all_states'],
    verbose=True,
    linesearch=vars(args)['linsearch'],
    max_linesearch=vars(args)['maxlinsearch'],
    num_memories=vars(args)['nummem']
    )
    return model



def evaluate_raw_model_accuracy(X_dev, Y_dev, Y_dev_nonsimp, args):
    counter, missclassification_counter = 0, 0
    for example_i in range(len(X_dev)):
        for base_i in range(len(X_dev[example_i])):
            if str(max(['0','1','2','3','4'], key=X_dev[example_i][base_i].get))==str(Y_dev[example_i][base_i]):
                counter += 1
            else:
                counter += 1
                missclassification_counter += 1
    pooled_accuracy =  (counter-missclassification_counter)/counter
    print('Accuracy of model compared to pooled ground truths: '+str(pooled_accuracy))
    counter, missclassification_counter = 0, 0
    for example_i in range(len(X_dev)):
        for base_i in range(len(X_dev[example_i])):
            for extender_i in range(vars(args)['win']):
                if str(max(['0','1','2','3','4'], key=X_dev[example_i][base_i].get))==str(Y_dev_nonsimp[example_i][base_i*vars(args)['win']+extender_i]):
                    counter += 1
                else:
                    counter += 1
                    missclassification_counter += 1
    upsampled_accuracy = (counter-missclassification_counter)/counter
    print('Accuracy of model compared to un-pooled ground truths: '+str(upsampled_accuracy))
    
    
    
def evaluate_crf_accuracy(model, X_dev, Y_dev, Y_dev_nonsimp, args):
    Y_pred = model.predict(X_dev)
    counter, missclassification_counter = 0, 0
    for example_i in range(len(X_dev)):
        for base_i in range(len(X_dev[example_i])):
            if str(Y_pred[example_i][base_i])==str(Y_dev[example_i][base_i]):
                counter += 1
            else:
                counter += 1
                missclassification_counter += 1
    pooled_accuracy =  (counter-missclassification_counter)/counter
    print('Accuracy of CRF+model compared to pooled ground truths: '+str(pooled_accuracy))
    counter, missclassification_counter = 0, 0
    for example_i in range(len(X_dev)):
        for base_i in range(len(X_dev[example_i])):
            for extender_i in range(vars(args)['win']):
                if str(Y_pred[example_i][base_i])==str(Y_dev_nonsimp[example_i][base_i*vars(args)['win']+extender_i]):
                    counter += 1
                else:
                    counter += 1
                    missclassification_counter += 1
    upsampled_accuracy = (counter-missclassification_counter)/counter
    print('Accuracy of upsampled model evaluated against un-pooled ground truths: '+str(upsampled_accuracy))
    

def get_args():
    import argparse
    parser=argparse.ArgumentParser(description=_README)
    parser.add_argument('--outfile', metavar='output filename', type=str, required=False, 
                        default='new_model.pickle', help='output filename')
    parser.add_argument('--outpath', metavar='output path prefix', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/chain_CRF/stored_models/', help='output path prefix')
    parser.add_argument('--input_path', metavar='input data path prefix', type=str, required=False, 
                        default='/scratch/users/jsokol/deepLAI/CRF/chain_CRF/data/', help='input data path prefix')
    parser.add_argument('--it', metavar='1000', type=int, required=False, default=1000,
                         help='# of iterations to train for')
    parser.add_argument('--c1', metavar='0', type=float, required=False, default=0,
                         help='c1 regularizer (only relevant for lbfgs)')
    parser.add_argument('--c2', metavar='0', type=float, required=False, default=0,
                         help='c2 regularizer (only relevant for l2sgd, lbfgs)')
    parser.add_argument('--alg', metavar='lbfgs', type=str, required=False, default='lbfgs',
                         help='inference method (lbfgs, l2sgd, ap, pa or arow)')
    parser.add_argument('--all_trans', action='store_true',
                         help='generate all transition features')
    parser.add_argument('--all_states', action='store_true', 
                         help='generate all state features')
    parser.add_argument('--recprob', action='store_true', 
                         help='use recombination probabilities')
    parser.add_argument('--deep', action='store_true', 
                         help='use class probabilities of more neighboring examples')
    parser.add_argument('--win', metavar='1000', type=int, required=False, default=1000,
                         help='pooling window')
    parser.add_argument('--linsearch', metavar='Backtrack', type=str, required=False, default='Backtrack',
                         help='linsearch algorithm (MoreThuente, Backtracking or StrongBacktracking; only relevant for lbfgs)')
    parser.add_argument('--maxlinsearch', metavar='20', type=int, required=False, default=20,
                         help='max line search')
    parser.add_argument('--nummem', metavar='6', type=int, required=False, default=24,
                         help='number of limited memories')
    args=parser.parse_args()
    return args    



def main():
    args=get_args()
    print(args)
    X_train, Y_train, Y_train_nonsimp, X_dev, Y_dev, Y_dev_nonsimp = load_data(args)
    evaluate_raw_model_accuracy(X_dev, Y_dev, Y_dev_nonsimp, args)
    model = define_model(args)
    print('Starting fitting process')
    model.fit(X_train, Y_train, X_dev=X_dev, y_dev=Y_dev)
    evaluate_crf_accuracy(model, X_dev, Y_dev, Y_dev_nonsimp, args)
    pickle.dump(model, open(vars(args)['outpath']+vars(args)['outfile'],'wb'))
    


if __name__=='__main__':
    main()




