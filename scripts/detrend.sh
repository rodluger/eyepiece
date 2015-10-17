#!/bin/bash
#
# This script runs detrending for a given KOI several times, each with slightly
# different initial conditions. Call syntax:
#
# >>> ./detrend.sh KOI[17.01] NRUNS[10] BLIND[1]
#
#

python -c "import eyepiece; eyepiece.Inspect(koi = float(${1-17.01}), blind = bool(${3-1}))"
for i in $(seq 1 ${2-10})
do
    qsub -vTAG=$i,KOI=${1-17.01} detrend.pbs
done