#!/bin/bash
#
# This script runs detrending for a given KOI several times, each with slightly
# different initial conditions. Call syntax:
#
# >>> ./detrend.sh KOI[17.01] NRUNS[10] BLIND[1]
#
#

# Set up the directory
export PLDDIR=/usr/lusers/rodluger/src/templar/output/${1-17.01}/pld
rm -f $PLDDIR/*

# Submit the jobs
python -c "import eyepiece; eyepiece.Inspect(koi = float(${1-17.01}), blind = bool(${3-1}))"
for i in $(seq 1 ${2-10}); do
    qsub -vTAG=$i,KOI=${1-17.01} detrend.pbs > /dev/null
done

# Check for completion
success=0
sleep 60
for (( minute=0; minute < 120; ++minute )); do
    runs=0
    for i in $(seq 1 ${2-10}); do
        if cat $PLDDIR/pld_$i.log | grep -q "Detrending complete"; then
            runs=$((runs+1))
        fi
    done
    if (( runs == ${2-10})); then
        success=1
        break
    fi
    sleep 60
done

# Plot if successful
if ((success)); then
    python -c "import eyepiece; eyepiece.PlotDetrended(koi = float(${1-17.01}))"
fi