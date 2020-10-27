# Scripts for admixture simulation

Admixture simulation is performed separately in the dev and test sets (see [here](https://github.com/maguirre1/deepLAI/tree/master/data/reference-panel) for information on input data), with RFMix [simulate](https://github.com/slowkoni/rfmix).

### Types of simulation

Two types of admixture simulation is performed, with the number of generations `G` of mixture one of 3, 10, or 30. In each batch, `n=100` individuals are generated. As RFMix simulate requires reference LD information to determine realistic ancestry breakpoints, we use the recombination map from [HAPMAP v2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2689609/) -- `hapmap-phase2-genetic-map.tsv.gz`.

1. __Pan-continental admixture__: All individuals in the dev/test set are founders, and the population is grown at a rate of 1.5 up to a size of 500 before being downsampled down to the final `n=100`. Further information is in the script `simulate_set.sh`.

2. __Admixture between two groups__: Mixture between one major (`n=16` founders) and one minor (`n=4` founders) set of individuals. Grown at a rate of 1.5 up to a size of 500 before being sampled down to the final `n=100`. Further information is in the script `simulate_twopop.sh`.

### Experiment design

Finally, the script `simulate.sh` is a wrapper for each of the above scripts. It iterates through dev/test set, number of generations, and applicable pairs of populations for mixing. Note that smaller population sets (e.g. `OCE`) are too small to be the major population in the "two group" setup.

### Label conversion

By default, RFMix simulate outputs a tab delimited flat file indicating ancestral assignments. This file is converted to a numpy format analogous to the [genotype specification](../reference-panel/), with the script `label_convert_naive.py`. The script is "naive" in that it assumes the variants in the flat file are in the same order as the VCF for the reference panel.

RFMix simulate also outputs a VCF for each simulated individual. We also assume these data are ordered in the same way as in the population reference panel, and convert the simulated data to a numpy file format with the same [script](../reference-panel/vcf_to_numpy.py) used for the reference panel.
