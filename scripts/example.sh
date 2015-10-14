#!/bin/bash
for i in {1..24}
do
    qsub -vTAG=$i example.pbs
done