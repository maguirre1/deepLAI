#!/bin/bash

## input parameters -- ensuring usage

if [ $# -ne 4 ]; then
	echo "usage: bash $0 in_pop G pop1 pop2"
	exit 1
else
	in_pop=$1 # one of dev, test
	G=$2 # 3, 10, 30
	pop1=$3 # 80% sample: one of AFR AMR EAS EUR SAS
	pop2=$4 # 20% sample: see above
fi

# useful filenames
in_pop_vcf="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.${in_pop}.vcf"
sinfo="$(dirname $0)/../labels/igsr_samples_cleaned_version.tsv"
ref_map="$(dirname $0)/hapmap-phase2-recombination-map-21.tsv"
out="/home/magu/deepmix/data/ALL_DNA_dataset/simulated/${in_pop}_${pop1}_${pop2}_${G}gen" 



## make superpopulation map by subsampling individuals
superpop_map="${out}.superpop.map"

# 80%
grep -Fwf <(zgrep "#CHROM" $in_pop_vcf | head -1 | tr '\t' '\n' | tail -n +10) $sinfo \
	| awk -v p=$pop1 '($6==p){print $1"\t"$6}' \
        | shuf --random-source=<(echo "P0pul@t10nG3n3t1c$") -n 20 > $superpop_map

# 20%
grep -Fwf <(zgrep "#CHROM" $in_pop_vcf | head -1 | tr '\t' '\n' | tail -n +10) $sinfo \
	| awk -v p=$pop2 '($6==p){print $1"\t"$6}' \
	| shuf --random-source=<(echo "P0pul@t10nG3n3t1c$") -n 5 >> $superpop_map



## make subsampled VCF
in_vcf=${out}.input.vcf.b.gz

vcftools --vcf $in_pop_vcf --keep <(cut -f1 $superpop_map) --recode --stdout \
       | bgzip -c > $in_vcf

tabix $in_vcf



## simulation

# parameters
rate=1.5 # population growth rate
M=500 # maximum population size
N=100 # final sample size

# command
echo """/home/magu/deepmix/rfmix/simulate -f $in_vcf -g $ref_map -c 21 --random-seed=212 \
	-m $superpop_map --growth-rate=$rate --maximum-size=$M -s $N -G $G -o $out"""

/home/magu/deepmix/rfmix/simulate -f $in_vcf -g $ref_map -c 21 --random-seed=212 \
	-m $superpop_map --growth-rate=$rate --maximum-size=$M -s $N -G $G -o $out 

