# Data Overview

### Summary

We start with two files consisting of genotypes from individuals in 1000 Genomes, HGDP, and SGDP. One contains all individuals (n=3,558), and the other contains only non-admixed individuals (n=1,382). The latter non-admixed cohort was selected using ADMIXTURE. Details on the curation of these input data can be found in [this repository](https://github.com/rivas-lab/gsp/tree/master/reference_panel). In brief, the quality control procedure consists of these steps:

 - First, to remove indels
 - Second, to select only variants present in all three reference datasets (1KG, HDGP, and SGDP)
 - Third, to phase input genotypes for each individual
 - Fourth, to deduplicate samples
 - Fifth, to identify population structure using ADMIXTURE

Resulting population assignments, as well as sample info, are in the `reference_panel_metadata_w_qs.tsv` file in this folder. Relatedness information from 1KG is in the file `20130606_g1k.ped.txt`, and is considered when splitting the panel into train, dev, and test sets (see below). 



### File conversion

Scripts: 

1. `convert.sh`. A wrapper script which calls `vcf_to_numpy.py` (below) for each chromosome to prepare the reference panel (above) for input into the [segnet model](https://github.com/maguirre1/deepLAI/tree/master/model).

2. `vcf_to_numpy.py` converts phased sample genotypes in variant call format (VCF -- [specification here](https://samtools.github.io/hts-specs/VCFv4.2.pdf)) to a numpy genotype file, for use in deep learning models. See below for details on the specification of this file.


#### Specification for numpy genotype file

Prior to use in neural net models for LAI, genomes are processed into compressed numpy (npz) format. Each such file contains three matrixes -- one each for variant info, sample info, and the genotypes themselves. They are described below.

1. `V` is a variant info matrix of shape `(n_variants, 4)`. It corresponds to the first, second, fourth, and fifth rows of the input VCF. These are the chromosome, position, reference, and alternate allele(s) of that variant.

2. `S` is the sample info matrix, of shape `(2*n_samples,)`. We assume that the sample VCF is phased (with [BEAGLE](https://faculty.washington.edu/browning/beagle/beagle.html), or similar), hence there are `2n` haplotypes in the input. Each sample ID is taken as present in the VCF header, and appended with `_S1` for the first haplotype and `_S2` for the second.

3. `G` is the sample genotype matrix, of shape `(2*n_samples, n_variants, 2)`. Rows are sample haplotypes, columns are variants. Each entry is a one-hot encoded vector of length `2` indicating which of the alleles for a given variant is present on a given haplotype. The entire genotype matrix is of `dtype=bool`.


### Train/Dev/Test Split

Scripts: `filter_and_split.ipynb`. This jupyter notebook performs quality control: first, removing populations that are too small for learning (e.g. "African Hunter Gatherer" -- AHG); and second, ensuring that no familially related individuals are retained. It splits the resulting dataset (`n=1,345`) into training (`n=1,045`), dev (`n=100`), and test (`n=200`) sets. 

Lists of sample IDs for the train/dev/test sets are in the `split` subdirectory. For each population (e.g. `dev`) there are three files: (1) `dev.txt`, which containts sample IDs (2) `dev.superpop.txt`, which contains superpopulation labels (e.g. `EUR`) and sample IDs, and (3) `dev.strands.txt`, which contains "stranded" sample IDs (e.g. `NA12878_S1`). 

Extra subsets and transformations of sample lists have been made for a few different purposes (e.g. sorting individuals by population code, to standardize RFMix simulate results to a predicable output). We hope these files have been names in a similarly self-explanatory way.

