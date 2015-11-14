#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
classy.py
---------

A simple class wrapper for all the detrending functionality.

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Input, GetData
from . import download, preprocess, detrend, plot, compare
import os
import numpy as np
import matplotlib.pyplot as pl

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

class Eyepiece(object):
  '''
  
  '''
  
  def __init__(self, input_file = None):
  
    # Input file
    self.input_file = input_file
    self.inp = Input(self.input_file)
    
    # Save a copy of the input file (will overwrite!)
    datadir = os.path.join(self.inp.datadir, str(self.inp.id), '_data')
    if not os.path.exists(datadir):
      os.makedirs(datadir)
    with open(os.path.join(datadir, 'input.log'), 'w') as f:
      for key, val in sorted(zip(self.inp.__dict__.keys(), self.inp.__dict__.values())):
        if not key.startswith('_'):
          try:
            item = '%s: %s' % (key, str(val))
          except:
            item = '%s: ???' % (key)
          print(item, file = f)
    
    # Download the data
    download.DownloadData(self.inp.id, self.inp.dataset, long_cadence = self.inp.long_cadence, 
                          clobber = self.inp.clobber, datadir = self.inp.datadir, 
                          bad_bits = self.inp.bad_bits, aperture = self.inp.aperture, 
                          quarters = self.inp.quarters, quiet = self.inp.quiet,
                          inject = self.inp.inject, trninfo = self.inp.trninfo, pad = self.inp.padbkg,
                          ttvs = self.inp.ttvs)
    
    # Preprocess
    preprocess.Preprocess(self.input_file)
  
  def Detrend(self, pool = None):
    '''
    
    '''
    
    detrend.Detrend(self.input_file, pool = pool)
    detrend.ComputePLD(self.input_file)
  
  def Plot(self):
    '''
    
    '''

    plot.PlotTransits(self.input_file); pl.close()
    plot.PlotDetrended(self.input_file); pl.close()
    
    try:
      np.load(os.path.join(self.inp.datadir, str(self.inp.id), '_data', 'cmp.npz'))
      compare.PlotComparison(self.input_file); pl.close()
    except FileNotFoundError:
      # User hasn't run the comparison yet, so no big deal
      pass
    
    if not self.inp.quiet:
      print("Figures saved to %s." % os.path.join(self.inp.datadir, str(self.inp.id), '_plots'))
  
  def Compare(self, pool = None):
    '''
    
    '''
    
    compare.Compare(self.input_file, pool = pool)
  
  def WhiteData(self, folded = True, return_mean = False):
    '''
    
    '''
    
    return detrend.GetWhitenedData(self.input_file, folded = folded, return_mean = return_mean)
  
  def Data(self, data_type = 'trn'):
    '''
    
    '''
    
    return GetData(self.inp.id, data_type = data_type, datadir = self.inp.datadir)