#!/bin/bash

## input parameters -- ensuring usage

if [ $# -ne 2 ]; then
	echo "usage: bash $0 in_pop n_generations"
	exit 1
else
	in_pop=$1 # one of dev, test
	G=$2 # 3, 10, 30
fi

# useful filenames
in_pop_vcf="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.${in_pop}.vcf"
in_pop_map="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.${in_pop}.superpop.txt"
sinfo="$(dirname $0)/../labels/igsr_samples_cleaned_version.tsv"
ref_map="$(dirname $0)/hapmap-phase2-recombination-map-21.tsv"
out="/home/magu/deepmix/data/ALL_DNA_dataset/simulated/${in_pop}_${G}gen" 


# parameters
rate=1.5 # population growth rate
M=500 # maximum population size
N=100 # final sample size

# command
/home/magu/deepmix/rfmix/simulate -f $in_pop_vcf -g $ref_map -c 21 --random-seed=212 \
	-m $in_pop_map --growth-rate=$rate --maximum-size=$M -s $N -G $G -o $out 

