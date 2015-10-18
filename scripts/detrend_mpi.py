#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from eyepiece.mpi_pool import MPIPool
from eyepiece import Inspect, GetData, Detrend, PlotDetrended

pool = MPIPool()

if not pool.is_master():
  pool.wait()
  sys.exit(0)

# USER
koi = 17.01
niter = 5

# Try to load the data
try:
  GetData(koi = koi, data_type = 'bkg')
except:
  Inspect(koi = koi, blind = True)

# Detrend and plot
Detrend(koi = koi, pool = pool, tags = range(niter))
PlotDetrended(koi = koi)

pool.close()