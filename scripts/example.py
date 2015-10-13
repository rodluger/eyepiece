#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from eyepiece.pld import Run, Plot
from eyepiece.interruptible_pool import InterruptiblePool
import sys

# User
koi = 254.01
quarters = [3]
niter = 2
maxfun = 50
debug = True
pool = InterruptiblePool()


# DEBUG
pool = None
# DEBUG


# Tag number
try:
  tag = sys.argv[1]
except:
  tag = 0

# Run and plot
Run(koi = koi, quarters = quarters, tag = tag, maxfun = maxfun, pool = pool, debug = debug)
Plot(koi = koi, quarters = quarters)