#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from eyepiece.pld import Run, Plot

koi = 254.01
quarters = list(range(1,18))
niter = 2
maxfun = 50
debug = True

try:
  from mpi4py import MPI
  from eyepiece.mpi_pool import MPIPool
  multi = 'mpi'
except:
  try:
    from multiprocessing.pool import Pool
    multi = 'mp'
  except:
    multi = 'none'
  
if multi == 'mpi':
  print("Parallelizing with MPI.")
  pool = MPIPool(loadbalance = True)
  if not pool.is_master():
    pool.wait()                                            
    sys.exit(0)
elif multi == 'mp':
  print("Parallelizing with multiprocessing.")
  pool = Pool()
else:
  print("No parallelization.")
  pool = None

# Run and plot
Run(koi = koi, quarters = quarters, niter = niter, maxfun = maxfun, pool = pool, debug = debug)
Plot(koi = koi, quarters = quarters)

# Close
if multi == 'mpi':
  pool.close()