#!/bin/bash
for i in {1..16}
do
    qsub -vTAG=$i example.pbs
done