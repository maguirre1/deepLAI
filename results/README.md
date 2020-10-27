# Evaluating LAI models

This directory contains results from evaluating our [SegNet model](../model) and an RFMix [baseline model](../baseline). Results from three experiments are here:

1. SegNet hyperparameter optimization (this is coming soon!)

2. SegNet test set results (`evaluate.py` is the generator script; results are packaged as `segnet_results.tar.gz`). This contains text files containing LaTeX-formatted confusion matrices for a series of test sets with varied ancestry compositions (see info on [admixture simulation](../data/admixture-sim/)). Example output (pretty-printed in a jupyter notebook) is in `evaluate.ipynb`.

3. RFMix test set results (`evaluate_rfmix.py` is the generator script; results are packaged as `rfmix evaluations.zip`). This is analogous to results for SegNet, with examples (also pretty-printed) in `evaluate_rfmix.ipynb`).

A more detailed description of results is forthcoming, and will be included both here and in preprint (check the main repository page for updates!).
