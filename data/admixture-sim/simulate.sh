#!/bin/bash

# iterate over evaluation set, mixing time, populations
for pop in "dev" "test"; do
    for G in 3 10 30; do
        # expand entire set
        bash simulate_set.sh $pop $G
	if [[ $pop == "dev" ]]; then continue; fi
	# expand targeted subsample in test set (larger)
        for p1 in "AFR" "EAS" "EUR" "NAT" "SAS"; do # these have test n >= 16 
            for p2 in "AFR" "EAS" "EUR" "NAT" "OCE" "SAS" "WAS"; do # n >= 4
		# don't mix with yourself
		if [[ $p1 == $p2 ]]; then continue; fi
		# expand targeted subsample
                bash simulate_twopops.sh $pop $G $p1 $p2
            done
        done
    done
done
