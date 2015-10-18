#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from eyepiece.mpi_pool import MPIPool
from eyepiece import Detrend, PlotDetrended

# Set up MPI
pool = MPIPool()
if not pool.is_master():
  pool.wait()
  sys.exit(0)

# Detrend and plot
Detrend(koi = 17.01, pool = pool, tags = range(5))
PlotDetrended(koi = 17.01)

# Close the pool
pool.close()