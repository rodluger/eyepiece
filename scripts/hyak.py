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

>>> mpi hyak.py input.py

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from eyepiece import Eyepiece
import para
import sys

# Did the user provide an input file?
try:
  inpfile = str(sys.argv[1])
except:
  raise ValueError('Please provide a valid input file.')

# Create a ``para`` MPI pool instance
pool = para.Pool(loadbalance = True)

# Grab the data
eye = Eyepiece(inpfile)

# Detrend
eye.Detrend(pool = pool)

# Compare to other methods
eye.Compare(pool = pool)

# Release the nodes
pool.close()

# Plot the results
eye.Plot()