#!/bin/bash
#SBATCH -J convert
#SBATCH --mem=32000
#SBATCH -t 3:00:00
#SBATCH -o logs/convert.%A_%a.out
#SBATCH --array=1-22


# start with this file and load this software
in_vcf="/oak/stanford/groups/mrivas/public_data/ref_1kg_hgdp_sgdp/beagle_1kg_hgdp_sgdp_ref_panel_pure.vcf.gz"
ml load biology; ml load bcftools; ml load htslib # bcftools, bgzip, tabix


# consider this chromosome
chm=${SLURM_ARRAY_TASK_ID:=20}


# first make this vcf and tabix it just because
vcf2="$SCRATCH/deepmix/data/vcf/panel_chr${chm}.vcf.gz"
echo $vcf2
if [ ! -f $vcf2 ]; then # in case you have to rerun
	bcftools view -r "chr${chm}" $in_vcf | bgzip -c > $vcf2
	tabix $vcf2
fi

# now convert to numpy
npz="$SCRATCH/deepmix/data/panel_chr${chm}"
echo $npz
python vcf_to_numpy.py $vcf2 $npz
