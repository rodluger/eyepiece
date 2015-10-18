#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from eyepiece.mpi_pool import MPIPool
from eyepiece import Detrend

pool = MPIPool()

if not pool.is_master():
  pool.wait()
  sys.exit(0)

Detrend(koi = 17.01, pool = pool, tags = range(10))

pool.close()