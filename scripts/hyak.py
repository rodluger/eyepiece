#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
hyak.py
-------

Runs ``eyepiece``. Call this script with ``mpi`` for
MPI parallelization on Hyak. Note that the raw data must
be cached prior to submitting the job. Simply run

>>> python -c "import eyepiece; eyepiece.Eyepiece('input.py')"

before executing this script with ``mpi``:

>>> mpi hyak.py -i input.py

'''

import matplotlib as mpl; mpl.use('Agg')
from eyepiece import Eyepiece
import para

# Create a ``para`` MPI pool instance
pool = para.Pool(loadbalance = True)

# Grab the data
eye = Eyepiece('input.py')

# Detrend
eye.Detrend(pool = pool)

# Compare to other methods
eye.Compare(pool = pool)

# Release the nodes
pool.close()

# Plot the results
eye.Plot()