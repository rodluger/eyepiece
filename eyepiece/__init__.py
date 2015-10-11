#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
from .inspect import *

# Info
__version__ = "0.0.1"
__author__ = "Rodrigo Luger (rodluger@uw.edu)"
__copyright__ = "Copyright 2015 Rodrigo Luger"

# Disable MPL keyboard shortcuts
pl.rcParams['toolbar'] = 'None'
pl.rcParams['keymap.all_axes'] = ''
pl.rcParams['keymap.back'] = ''
pl.rcParams['keymap.forward'] = ''
pl.rcParams['keymap.fullscreen'] = ''
pl.rcParams['keymap.grid'] = ''
pl.rcParams['keymap.home'] = ''
pl.rcParams['keymap.pan'] = ''
pl.rcParams['keymap.quit'] = ''
pl.rcParams['keymap.save'] = ''
pl.rcParams['keymap.xscale'] = ''
pl.rcParams['keymap.yscale'] = ''
pl.rcParams['keymap.zoom'] = ''

# Kepler cadences
KEPLONGEXP =              (1765.5/86400.)
KEPSHRTEXP =              (58.89/86400.)