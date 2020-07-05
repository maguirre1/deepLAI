#!/bin/bash

## input parameters -- ensuring usage

if [ $# -ne 5 ]; then
	echo "usage: bash $0 in_pop G pop1 pop2 chrom"
	exit 1
else
	in_pop=$1 # one of dev, test
	G=$2      # 3, 10, 30
	pop1=$3   # 80% sample: one of AFR EAS EUR NAT SAS
	pop2=$4   # 20% sample: one of AFR EAS EUR NAT OCE SAS WAS
	c=$5  # chromosome
fi

# useful filenames
in_pop_vcf="/home/magu/deepmix/data/reference_panel/vcf/panel.${in_pop}.vcf.gz"
sinfo="../reference-panel/split/${in_pop}.superpop.txt"
ref_map="$(dirname $0)/hapmap-phase2-genetic-map.tsv"
out="/home/magu/deepmix/data/reference_panel/simulated_chr${c}/vcf/${in_pop}_${pop1}_${pop2}_${G}gen"

# make sure the output is legal
mkdir -p $( dirname $out )



## make superpopulation map by subsampling individuals
superpop_map="${out}.superpop.map"

# 80%
awk -v p=$pop1 '$2==p' $sinfo | shuf -n 16 --random-source=$sinfo > $superpop_map

# 20%
awk -v p=$pop2 '$2==p' $sinfo | shuf -n 4 --random-source=$sinfo >> $superpop_map



## make subsampled VCF
in_vcf=${out}.input.vcf.b.gz

bcftools view -Oz -S <( awk '{print $1}' $superpop_map ) $in_pop_vcf > $in_vcf

tabix $in_vcf



## simulation

# parameters
rate=1.5 # population growth rate
M=500 # maximum population size
N=100 # final sample size

# command
/home/magu/deepmix/rfmix/simulate -f $in_vcf -g $ref_map -c $c --random-seed=$c \
	-m $superpop_map --growth-rate=$rate --maximum-size=$M -s $N -G $G -o $out 

