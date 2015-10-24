#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

Runs full detrending on a given KOI.

>>> python example.py /path/to/input/script.py

'''

import subprocess
from eyepiece.inspect import Inspect
from eyepiece.download import GetData
from eyepiece.utils import Input
from eyepiece.detrend import Detrend, PlotDetrended, PlotTransits
import os
import sys

if __name__ == '__main__':
  
  try:
    input_file = os.path.abspath(str(sys.argv[1]))
    inp = Input(input_file)
  except (IndexError, FileNotFoundError):
    raise Exception("Please provide a valid input file!")
    
  # Try to load the data. If it fails, run ``Inspect``
  try:
    GetData(koi = inp.koi, data_type = 'bkg', datadir = inp.datadir)
  except Exception:
    success = Inspect(input_file)
    if not success:
      if not inp.quiet:
        print("Detrending aborted!")
      sys.exit()
  
  # Run detrending. You can optionally specify a parallelization pool instance here.
  Detrend(input_file = input_file, pool = None)
  
  # Plot the results
  PlotDetrended(input_file = input_file)
  PlotTransits(input_file = input_file)