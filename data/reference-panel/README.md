# Data Overview

### Summary

We start with two files consisting of genotypes from individuals in 1000 Genomes, HGDP, and SGDP. One contains all individuals (n=3,558), and the other contains only non-admixed individuals (n=1,380). The latter non-admixed cohort was selected using ADMIXTURE. Details on the curation of these input data can be found in [this repository](https://github.com/rivas-lab/gsp/tree/master/reference_panel). In brief, the quality control procedure consists of these steps:

 - First, to remove indels
 - Second, to select only variants present in all three reference datasets (1KG, HDGP, and SGDP)
 - Third, to phase input genotypes for each individual
 - Fourth, to deduplicate samples
 - Fifth, to identify population structure using ADMIXTURE

Resulting population assignments, as well as sample info, are in the `reference_panel_metadata_w_qs.tsv` file in this folder. Relatedness information from 1KG is in the file `20130606_g1k.ped.txt`, and is considered when splitting the panel into train, dev, and test sets (see below). Genotype data for the entire panel and non-admixed subset can be found here, on Sherlock:

```
# Entire panel (n=3558) -- positions in hg38 for both files
/oak/stanford/groups/mrivas/public_data/ref_1kg_hgdp_sgdp/beagle_1kg_hgdp_sgdp_ref_panel.vcf.gz

# Non-admixed set (n=1380)
/oak/stanford/groups/mrivas/public_data/ref_1kg_hgdp_sgdp/beagle_1kg_hgdp_sgdp_ref_panel_pure.vcf.gz 
```



### File conversion

Scripts: `convert.sh`, `vcf_to_numpy.py`, `vcf_to_this_numpy.py`



### Train/Dev/Test Split

Scripts: `split_pop.py`

