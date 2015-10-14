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
simple = True

# Tag number
try:
  tag = sys.argv[1]
  if tag == '-p':
    Plot(koi = koi, quarters = quarters)
    sys.exit()
  else:
    tag = int(tag)
except IndexError:
  tag = 0

# Run!
Run(koi = koi, quarters = quarters, tag = tag, pool = pool, simple = simple)