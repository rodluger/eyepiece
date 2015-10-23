#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
plot.py
-------

>>> python plot.py /path/to/input/script.py

'''

import matplotlib; matplotlib.use('Agg')
import sys
import os
from eyepiece.detrend import PlotDetrended, PlotTransits

if __name__ == '__main__':

  # Grab input file from args
  input_file = os.path.abspath(str(sys.argv[1]))
  
  # Plot
  PlotDetrended(input_file = input_file)
  PlotTransits(input_file = input_file)