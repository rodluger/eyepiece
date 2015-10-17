#!/bin/bash
#
# This script plots the result of the detrending for a given KOI.
#
# >>> ./plot.sh KOI[17.01]
#
#

python -c "import eyepiece; eyepiece.PlotDetrended(koi = float(${1-17.01}))"