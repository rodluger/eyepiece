#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import matplotlib; matplotlib.use('Agg')
from eyepiece.pld import Run, Plot
from eyepiece.interruptible_pool import InterruptiblePool
import sys

# User
koi = 254.01
quarters = list(range(1,18))
pool = InterruptiblePool()

# Tag number
try:
  tag = int(sys.argv[1])
except:
  tag = 0

# Run and plot
Run(koi = koi, quarters = quarters, tag = tag, pool = pool)
Plot(koi = koi, quarters = quarters)