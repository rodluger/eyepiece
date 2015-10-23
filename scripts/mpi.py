#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
mpi.py
------

>>> python mpi.py /path/to/input/script.py

'''

import matplotlib; matplotlib.use('Agg')
import sys
import os
from eyepiece.mpi_pool import MPIPool
from eyepiece.detrend import Detrend

if __name__ == '__main__':

  # Grab input file from args
  input_file = os.path.abspath(str(sys.argv[1]))

  # Set up MPI
  pool = MPIPool()
  if not pool.is_master():
    pool.wait()
    sys.exit(0)

  # Detrend
  Detrend(input_file = input_file, pool = pool)

  # Close the pool
  pool.close()