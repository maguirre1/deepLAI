#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from os import path


_README="""
A script to generate confusion matrix and evaluate accuracy of rfmix.

-Matthew Aguirre (magu[at]stanford[dot]edu)
"""


_TODO="""
1. modify output directory (currently the output text file is stored in the directory that this script is called from). 
"""


# quick check if we're on galangal
import platform
if platform.uname()[1]=='galangal.stanford.edu':
    print('error: script must be modified to be run on galangal')
else: 
    # assume we're on sherlock -- load modules and check versions
    print('Assuming we are on sherlock')

    

# define functions
def load_data(args_dict):
    print('Loading in data')
    ## check if output file already exists
    if path.exists(args_dict['output-filename']):
        print('Error: output file already exists. Aborting script.')
        return '', '', True
    ## load y and yhat_raw
    data_dir='/scratch/users/magu/deepmix/data/simulated_chr20/'
    yhat_raw=pd.read_table(data_dir+'vcf/rf_out/'+args_dict['rfmix-result-filename'], skiprows=1)
    y=np.load(data_dir+'label/'+args_dict['gt-filename'])
    
    return y, yhat_raw, False
    
    

def expand_rfmix_windows(y, yhat_raw, S):
    print('Expanding rfmix windows')
    V_pos=y['V'][:,1].astype(int)
    yhat=pd.DataFrame(index=['_'.join(s) for s in y['V']], columns=S)
    
    for ix in range(yhat_raw.shape[0]):
        ids=(yhat_raw.iloc[ix,1] <= V_pos) & (V_pos <= yhat_raw.iloc[ix,2])
        yhat.iloc[ids,:]=np.vstack([yhat_raw.iloc[ix,6:] for _ in range(sum(ids))]).astype(int)+1
        
    return yhat

    

def evaluate_model(y, yhat, args_dict):
    print('Evaluating model and creating text file')
    ## create df of confusion matrices and evaluate accuracy
    # confusion
    cm=confusion_matrix(y['L'].flatten(), yhat.T.values.flatten().astype(int))
    # accuracy
    acc=np.sum(np.diag(cm))/np.sum(cm)
    anc_label=['AFR', 'EAS', 'EUR', 'NAT', 'SAS']
    row_normalized_df = pd.DataFrame(cm, index=anc_label, columns=anc_label).divide(cm.sum(axis=1), axis=0)
    col_normalized_df = pd.DataFrame(cm, index=anc_label, columns=anc_label).divide(cm.sum(axis=0), axis=1)
    bp_df = pd.DataFrame(cm, index=anc_label, columns=anc_label)
    
    ## write df and accuracy to text file
    output_file_handle = open(args_dict['output-filename'],"w") 
    output_file_handle.writelines('Row-normalized confusion matrix:\n\n')
    output_file_handle.writelines(row_normalized_df.to_string())
    output_file_handle.writelines('\n\n\n\nCol-normalized confusion matrix:\n\n')
    output_file_handle.writelines(col_normalized_df.to_string())
    output_file_handle.writelines('\n\n\n\nBP confusion matrix:\n\n')
    output_file_handle.writelines(bp_df.to_string())
    output_file_handle.writelines('\n\n\n\nmodel accuracy = '+str(acc))
    output_file_handle.close()
    
    

def get_args():
    import argparse
    parser=argparse.ArgumentParser(description=_README)
    parser.add_argument('rfmix-result-filename', metavar='rfmix_result', type=str,
                         help='filename of rfmix output (.msp.tsv file)')
    parser.add_argument('gt-filename', metavar='groundtruth', type=str,
                         help='filename of ground truth labels (.npz file)')
    parser.add_argument('output-filename', metavar='output', type=str,
                         help='output filename (.txt file)')
    args=parser.parse_args()
    return args



def main():
    args=get_args()
#    print(args)
    y, yhat_raw, abort = load_data(vars(args))
    if abort: return
    S = np.array([s.replace('_S1','.0').replace('_S2','.1') for s in y['S']]) # match samples
    yhat = expand_rfmix_windows(y, yhat_raw, S) # expand rfmix windows
    evaluate_model(y, yhat, vars(args)) # evaluate model and save accuracy & confusion matrices to text file
    return
    


if __name__=='__main__':
    main()




