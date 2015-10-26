#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
plot.py
-------

Plot the results of a detrending run.

>>> python plot.py /path/to/input/script.py

'''

import matplotlib; matplotlib.use('Agg')
import sys
import os
from eyepiece.detrend import PlotDetrended, PlotTransits
from eyepiece.utils import Input

if __name__ == '__main__':

  # Did the user specify an input file?
  if len(sys.argv) == 2:
    input_file = os.path.abspath(str(sys.argv[1]))
  else:
    # Assume it's in the cwd
    input_file = 'input.py'
  # Let's try to load it
  try:
    inp = Input(input_file)
  except:
    raise Exception("Please provide a valid input file!")
  
  # Plot
  PlotDetrended(input_file = input_file)
  PlotTransits(input_file = input_file)