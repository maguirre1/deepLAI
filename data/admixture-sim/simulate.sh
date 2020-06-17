#!/bin/bash

# iterate over evaluation set, mixing time, populations
for pop in "dev" "test"; do
    for G in 3 10 30; do
        # expand entire set
        bash simulate_set.sh $pop $G
        for p1 in "AFR" "EAS" "EUR" "SAS"; do 
            for p2 in "AFR" "EAS" "EUR" "SAS"; do
		# don't mix with yourself
		if [[ $p1 == $p2 ]]; then continue; fi
		# expand targeted subsample
                bash simulate_twopops.sh $pop $G $p1 $p2
            done
        done
    done
done
