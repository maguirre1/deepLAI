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
if [ $(hostname) = "galangal.stanford.edu" ]; then 
	traing="/home/magu/deepmix/data/reference_panel/vcf/panel_chr20.no-OCE-WAS.vcf.gz"
else
	traing="/scratch/users/magu/deepmix/data/vcf/panel_chr20.no-OCE-WAS.vcf.gz"
	alias rfmix "~/miniconda3/envs/popgen/bin/rfmix"
fi
labels="../data/reference-panel/split/train.superpop.sorted.txt"
ref_ld="../data/admixture-sim/hapmap-phase2-genetic-map.tsv.gz"


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
rfmix -f $in_vcf -r $traing -m $labels -g <( zcat $ref_ld | awk '$1==20') \
      -o $output --n-threads=12 --chromosome=20
