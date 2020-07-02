#!/bin/bash

## input parameters -- ensuring usage

if [ $# -ne 2 ]; then
	echo "usage: bash $0 in_pop n_generations"
	exit 1
else
	in_pop=$1 # one of dev, test
	G=$2      # 3, 10, 30
fi

# useful filenames
in_pop_vcf="/home/magu/deepmix/data/reference_panel/vcf/panel.${in_pop}.vcf.gz"
in_pop_map="$(dirname $0)/../reference-panel/split/${in_pop}.superpop.txt"
ref_map="$(dirname $0)/hapmap-phase2-genetic-map.tsv"
out="/home/magu/deepmix/data/reference_panel/simulated_chr20/vcf/${in_pop}_${G}gen" 

# make sure out is a legal filename
mkdir -p $( dirname $out )


# parameters
rate=1.5 # population growth rate
M=500 # maximum population size
N=100 # final sample size
c=20  # chromosome -- make sure this matches ref_map above

# sort the superpop labels, so the ground truth numbering is easier to deal with
in_map="${out}.superpop.map.txt"
awk 'BEGIN{OFS="\t"}{print $1,$2}' $in_pop_map | sort -k2,2 -k1,1 > $in_map

# command
/home/magu/deepmix/rfmix/simulate -f $in_pop_vcf -g $ref_map -c $c --random-seed=$c \
	-m $in_map --growth-rate=$rate --maximum-size=$M -s $N -G $G -o $out 

