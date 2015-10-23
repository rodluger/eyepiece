#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
lnlike.py
---------

>>> python launcher.py /path/to/input/script.py

'''

import matplotlib; matplotlib.use('Agg')
import sys
import os
from eyepiece.mpi_pool import MPIPool
from eyepiece.detrend import Detrend, PlotDetrended

if __name__ == '__main__':

  # Grab input file from args
  input_file = os.path.abspath(str(sys.argv[1]))

  # Set up MPI
  pool = MPIPool()
  if not pool.is_master():
    pool.wait()
    sys.exit(0)

  # Detrend and plot
  Detrend(input_file = input_file, pool = pool)
  PlotDetrended(input_file)
  PlotTransits(input_file)

  # Close the pool
  pool.close()