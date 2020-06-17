#!/bin/bash

# this script does a (prespecified) test/train/split on a vcf using vcftools


# we're going to be breaking up this vcf
in_vcf="/home/magu/deepmix/data/ALL_DNA_dataset/chm21.vcf.gz"


# loop over lists of individuals
for f in $(ls /home/magu/deepmix/data/ALL_DNA_dataset/*.inds.txt | sort); do 
	
	# get name of subgroup (dev, test, train) and make new filename
	pop=$(basename $f | awk -F'.' '{print $2}')
	out=$(echo $in_vcf | awk -v p=$pop '{gsub(".vcf.gz","."p); print}')
	
	# subsample and output to new file
	vcftools --gzvcf $in_vcf --keep $f --recode --out $out

done
