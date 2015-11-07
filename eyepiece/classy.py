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

class Eyepiece(object):
  '''
  
  '''
  
  def __init__(self, input_file = None):
  
    # Input file
    self.input_file = input_file
    self.inp = Input(self.input_file)
    
    # Download the data
    download.DownloadData(self.inp.id, self.inp.dataset, long_cadence = self.inp.long_cadence, 
                          clobber = self.inp.clobber, datadir = self.inp.datadir, 
                          bad_bits = self.inp.bad_bits, aperture = self.inp.aperture, 
                          quarters = self.inp.quarters, quiet = self.inp.quiet)
    
    # Preprocess
    preprocess.Preprocess(self.input_file)
  
  def Detrend(self, pool = None):
    '''
    
    '''
    
    detrend.Detrend(self.input_file, pool = pool)
  
  def Plot(self):
    '''
    
    '''
    
    plot.PlotDetrended(self.input_file)
    plot.PlotTransits(self.input_file)
    
    if not self.inp.quiet:
      print("Figures saved to %s." % os.path.join(self.inp.datadir, str(self.inp.id), '_plots'))
  
  def Compare(self):
    '''
    
    '''
    
    compare.PlotComparison(self.input_file)
    
    if not self.inp.quiet:
      print("Figure saved to %s." % os.path.join(self.inp.datadir, str(self.inp.id), '_plots'))

  def WhiteData(self, folded = True, return_mean = False):
    '''
    
    '''
    
    return detrend.GetWhitenedData(self.input_file, folded = folded, return_mean = return_mean)
  
  def Data(self, data_type = 'trn'):
    '''
    
    '''
    
    return GetData(self.inp.id, data_type = data_type, datadir = self.inp.datadir)