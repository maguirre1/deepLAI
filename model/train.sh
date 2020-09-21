#!/bin/bash
#SBATCH -J segnet
#SBATCH -p gpu,owners
#SBATCH --gpus=1 
#SBATCH -C GPU_MEM:32GB
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=32G
#SBATCH --out logs/array.hparam.%A.%a.out
# #SBATCH --array=93-180 # for models with 2e5 to 2e6 parameters
# #SBATCH --array=52-101 # for models with 5e4 to 5e5 params, index 52-127

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
  ml load python/3.6.1 cuda/10.2.89 cudnn/7.6.5; # openmpi/4.0.3
  ml load py-numpy/1.18.1_py36 py-scipy/1.4.1_py36 py-tensorflow/2.1.0_py36;
  # display node info and packages
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  which python3
  ml list
fi


id=${SLURM_ARRAY_TASK_ID:=20}
fs="$( awk -v nr=$id '(NR==nr){print $1}' hparam_to_nparam.tsv )"
nf="$( awk -v nr=$id '(NR==nr){print $2}' hparam_to_nparam.tsv )"
nb="$( awk -v nr=$id '(NR==nr){print $4}' hparam_to_nparam.tsv )"


mkdir -p weights
# call the model training script
python3 train.py --out weights/chr20.full.${id} --chrom=20 --num-epochs=200 \
  --filter-size=$fs --num-filter=$nf --num-blocks=$nb --batch-size=4 --continue-train

 
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
