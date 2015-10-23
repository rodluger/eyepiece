#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
defaults.py
-----------

These are the default config values used by ``eyepiece``. You can override these 
by creating your own ``input.py`` file and setting custom values for these parameters.

'''

# -------
# IMPORTS
# -------

import george

# ------
# PARAMS
# ------

# A
aperture = 'optimal'

# B
bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17]

# C
clobber = False

# D
datadir = '/Users/rodrigo/src/eyepiece/lightcurves'
debug = False
detrend_figsize = (48, 16)
dt_tol = 0.5

# I
interactive_inspect = True
interactive_detrend = False

# K
kbounds = [[1.e-8, 1.e8], [1.e-4, 1.e8]]
kinit = [100., 100.]
kernel = 1. * george.kernels.Matern32Kernel(1.)
koi = 17.01

# L
long_cadence = True

# M
maxfun = 15000
min_sz = 300

# P
padbkg = 2.0
padtrn = 5.0
pert_sigma = 0.25

# Q
quarters = range(18)
quiet = False

# S
split_cads = [4472, 6717]

# T
ttvpath = '/Users/rodrigo/src/templar/ttvs'
ttvs = False

# -------
# CLEANUP
# -------

del george

# ----
# DOCS
# ----

class _Docs(object):
  '''
  
  '''
  
  def __init__(self):
    
    # A
    self.aperture = 'Which aperture to use? ``optimal`` or ``full``'
    
    # B
    self.bad_bits = 'Kepler bad bit flags'
    
    # C
    self.clobber = 'Overwrite existing data?'
    
    # D
    self.datadir = 'Directory to store lightcurve processing data in'
    self.debug = 'Debug mode?'
    self.detrend_figsize = 'Size of figure returned by ``PlotDetrend`` in inches'
    self.dt_tol = 'Transit gap tolerance in days'
    
    # I
    self.interactive_inspect = 'Process data in interactive mode when inspecting?'
    self.interactive_detrend = 'Process data in interactive mode when detrending?'
    
    # K
    self.kbounds = 'Bounds on the kernel parameters'
    self.kinit = 'Initial values for the kernel parameters'
    self.kernel = 'The kernel to use for GP detrending'
    self.koi = 'Kepler KOI number'
    
    # L
    self.long_cadence = 'Use Kepler long cadence data?'
    
    # M
    self.maxfun = 'Maximum number of ln-like function calls in ``l_bfgs_b``'
    self.min_sz = 'Minimum chunk size in cadences when splitting the lightcurve'
    
    # P
    self.padbkg = 'Padding in units of the transit duration for masking transits in background data'
    self.padtrn = 'Padding in units of the transit duration for selecting transits for the transit-only data'
    self.pert_sigma = 'Perturb the initial conditions by this many sigma'
    
    # Q
    self.quarters = 'List of quarters to process (if data is present)'
    self.quiet = 'Suppress output?'
    
    # S
    self.split_cads = 'Cadences at which to split the data'
    
    # T
    self.ttvpath = 'Path to folder containing KOI ttv information'
    self.ttvs = 'Analyze this system assuming ttvs?'

