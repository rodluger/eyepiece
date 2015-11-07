#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ['Help', 'Eyepiece']

# Try to use TkAgg backend, and issue warning if it didn't work
import matplotlib; matplotlib.use('TkAgg', warn = False)
if matplotlib.get_backend() != 'TkAgg':
  print("WARNING: Unable to load TkAgg backend. Interactive preprocessing may not work correctly.")

# Import local files
from . import detrend, download, linalg, defaults, plot, preprocess, utils, compare, classy
from .utils import Help
from .classy import Eyepiece

# Info
__version__ = "0.0.1"
__author__ = "Rodrigo Luger (rodluger@uw.edu)"
__copyright__ = "Copyright 2015 Rodrigo Luger"