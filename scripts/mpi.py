#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib; matplotlib.use('Agg')
import sys
from eyepiece.mpi_pool import MPIPool
from eyepiece.detrend import Detrend, PlotDetrended

if __name__ == '__main__':

  # Grab input file from args
  try:
    input_file = os.path.abspath(str(sys.argv[1]))
  except:
    input_file = None

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