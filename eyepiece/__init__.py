#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import matplotlib; matplotlib.use('TkAgg', warn = False)
import matplotlib.pyplot as pl

from . import (detrend, download, lnlike, defaults, 
               inspect, mpi_pool, interruptible_pool,
               utils)

from .utils import Help

# Info
__version__ = "0.0.1"
__author__ = "Rodrigo Luger (rodluger@uw.edu)"
__copyright__ = "Copyright 2015 Rodrigo Luger"

# Kepler cadences
KEPLONGEXP =              (1765.5/86400.)
KEPSHRTEXP =              (58.89/86400.)