#!/bin/bash
#SBATCH -J segnet
#SBATCH -p gpu,owners
#SBATCH --gpus=1 
#SBATCH -C GPU_MEM:32GB
#SBATCH -t 12:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=32G
#SBATCH --out logs/array_%A.out

__README="""
A wrapper script for train.py
"""


# first check if we're on galangal
hn=$(hostname)
if [ $hn = "galangal.stanford.edu" ]; then 
  nvidia-smi
else
  # if not then fiddle around with modules
  ml purge; 
  ml load python/3.6.1 cuda/10.1.168 cudnn/7.4.1.5; # openmpi/4.0.3
  ml load py-numpy/1.18.1_py36 py-scipy/1.4.1_py36 py-tensorflow/2.0.0_py36;
  # display node info
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  nvidia-smi
  # display packages
  which python3
  ml list
fi


# call the model training script
python3 train.py --out chr20_array --array-only \
  --filter-size=8 --num-filter=12 --num-blocks=4 --batch-size=128

 
PARAMS="""
  --chrom 20            Chromosome to use (must be in 1,2,...,22)
  --batch-size 4        Minibatch size for training
  --num-filters 8       Number of filters in first segnet layer
  --filter-size 16      Convolutional filter size in segnet
  --num-epochs 100      Number of epochs to train model
  --num-blocks 5        Number of down/upward blocks (equivalent to model
                        depth)
  --pool-size 4         Width of maxpool operator
  --dropout-rate 0.01   Dropout rate at each layer
  --input-dropout-rate 0.01
                        Dropout rate after input layer
  --batch-norm          Flag to use batch normalization
  --no-generator        Flag to not use generator object, and load all data
                        into memory
  --out model_weights   Output path prefix -- extensions automatically added
"""
