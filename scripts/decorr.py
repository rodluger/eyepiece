#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
decorr.py
---------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import matplotlib; matplotlib.use('Agg')
from eyepiece.detrend import Run, Plot
from eyepiece.interruptible_pool import InterruptiblePool
import sys

# User
koi = 17.01
pool = InterruptiblePool()

# Tag number
try:
  tag = sys.argv[1]
  if tag == '-p':
    Plot(koi = koi)
    sys.exit()
  else:
    tag = int(tag)
except IndexError:
  tag = 0

# Run!
Run(koi = koi, tag = tag, pool = pool)