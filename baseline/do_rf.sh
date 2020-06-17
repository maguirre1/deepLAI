#!/bin/bash


# handle input
if [ "$#" -lt 1 ]; then 
	echo "usage: bash do_rf.sh input.vcf [out_prefix]"
	exit 1
else
	in_vcf=$1
fi

# handle an optional second parameter (output prefix)
autout=$(basename $in_vcf | awk '{gsub(".vcf",""); gsub(".gz",""); print}')
if [ "$#" -gt 1 ]; then 
	output=$2
else 
	mkdir -p "$(dirname $in_vcf)/rf_out"
	output="$(dirname $in_vcf)/rf_out/${autout}"
fi


# these are fixed
traing="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.train.vcf"
labels="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.train.superpop.txt"
#traing="/home/magu/deepmix/data/ALL_DNA_dataset/vcfs/chm21_ALL_X.train.downsample.200.recode.vcf"
#labels="/home/magu/deepmix/data/ALL_DNA_dataset/vcfs/chm21_ALL_X.train.subsample200.map.tsv"
ref_ld="../admixture_sim/hapmap-phase2-recombination-map-21.tsv"


# sanity check:
which rfmix
echo """
input: $in_vcf
train set: $traing
train labels: $labels
reference panel: $ref_ld
out: $output

running:
rfmix -f $in_vcf -r $traing -m $labels -g $ref_ld -o $output 
"""


# do rfmix!
rfmix -f $in_vcf -r $traing -m $labels -g $ref_ld -o $output \
      --n-threads=12 --chromosome=21
