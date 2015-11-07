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
datadir = '/Users/rodrigo/Desktop/new/lightcurves'
dataset = 'Kepler'
debug = False
dt_tol = 0.5

# F
fullscreen = True

# I
id = 17.01
info = {}
interactive = True

# K
kbounds = [[1.e-8, 1.e8], [1.e-4, 1.e8]]
kernel = 1. * george.kernels.Matern32Kernel(1.)
kinit = [100., 100.]

# L
long_cadence = True

# M
maxfun = 15000
min_sz = 300
mpn = 40

# N
niter = 5
nodes = 5

# P
padbkg = 2.0
padtrn = 5.0
pert_sigma = 0.25
plot_det_info = True
ppn = 12

# Q
quarters = range(18)
quiet = False

# S
split_cads = [4472, 6717]

# T
tbins = 20
ttvs = False

# W
walltime = 2

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
    self.dataset = 'Which exoplanet dataset to use (Kepler, K2, TESS...)'
    self.debug = 'Debug mode?'
    self.dt_tol = 'Transit gap tolerance in days'
    
    # F
    self.fullscreen = 'Fullscreen the plot when inspecting?'
    
    # I
    self.id = 'Kepler target ID (KOI number or KIC identifier)'
    self.info = 'Override the database values for the KOI\'s ``per``, ``tdur`` and the transit times ``tN`` by setting values for those in this dict'
    self.interactive = 'Process data in interactive mode?'
    
    # K
    self.kbounds = 'Bounds on the kernel parameters'
    self.kernel = 'The kernel to use for GP detrending'
    self.kinit = 'Initial values for the kernel parameters'
    
    # L
    self.long_cadence = 'Use Kepler long cadence data?'
    
    # M
    self.maxfun = 'Maximum number of ln-like function calls in ``l_bfgs_b``'
    self.min_sz = 'Minimum chunk size in cadences when splitting the lightcurve'
    self.mpn = 'Memory per node in GB when submitting a job with ``qsub``'
    
    # N
    self.niter = 'Number of iterations when detrending in parallel.'
    self.nodes = 'Number of nodes to use when submitting a job with ``qsub``'

    # P
    self.padbkg = 'Padding in units of the transit duration for masking transits in background data'
    self.padtrn = 'Padding in units of the transit duration for selecting transits for the transit-only data'
    self.pert_sigma = 'Perturb the initial conditions by this many sigma'
    self.plot_det_info = 'Plot detrending information?'
    self.ppn = 'Number of processors per node for a ``qsub`` job'
    
    # Q
    self.quarters = 'List of quarters to process (if data is present)'
    self.quiet = 'Suppress output?'
    
    # S
    self.split_cads = 'Cadences at which to split the data'
    
    # T
    self.tbins = 'Number of bins when plotting folded transits'
    self.ttvs = 'Analyze this system assuming ttvs?'
    
    # W
    self.walltime = 'Walltime in hours for a ``qsub`` job'