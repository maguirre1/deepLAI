#!/bin/bash
#SBATCH -J segnet
#SBATCH -p gpu,owners
#SBATCH --gpus=1
#SBATCH -t 20:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=64G


__README="""
A wrapper script for train.py
"""


# first check if we're on galangal
hn=$(hostname)
if [ $hn = "galangal.stanford.edu" ]; then 
	continue
else
	# if not then fiddle around with modules
	ml purge; 
	ml load python/3.6.1 cuda/10.1.168 cudnn/7.4.1.5;
	echo "load 1"
	ml load py-numpy/1.18.1_py36 py-tensorflow/2.0.0_py36;
	echo "load 2"	
	nvidia-smi --query-gpu=index,name --format=csv,noheader
	nvidia-smi
	which python3
	ml list
fi


# call the model training script
python3 train.py --out test_segnet_chr20
