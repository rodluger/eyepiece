#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
mpi.py
------

This script is meant to be called from within a PBS job. Call syntax:

>>> python mpi.py [/path/to/input/script.py]

'''

import matplotlib; matplotlib.use('Agg')
import sys
import os
from eyepiece.mpi_pool import MPIPool
from eyepiece.detrend import Detrend
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

  # Set up MPI
  pool = MPIPool()
  if not pool.is_master():
    pool.wait()
    sys.exit(0)

  # Detrend
  Detrend(input_file = input_file, pool = pool)

  # Close the pool
  pool.close()