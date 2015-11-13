#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
utils.py
--------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from . import defaults
import os
import subprocess
import numpy as np
import imp
import kplr
import random
import types

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

class FunctionWrapper(object):
  '''
  
  '''
  def __init__(self, f, *args, **kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, x):
    return self.f(x, *self.args, **self.kwargs)

def Input(input_file = None):
  '''
  
  '''
  
  if input_file is None:
    return defaults
  
  try:
    inp = imp.load_source("input", input_file) 
  except:
    raise IOError("Please provide a valid input file!")

  for key, val in list(inp.__dict__.items()):
    if key.startswith('_'):
      inp.__dict__.pop(key, None)
    else:
      # Check if user provided something they shouldn't have
      if key not in defaults.__dict__.keys() and not isinstance(inp.__dict__[key], types.ModuleType):                  
        raise Exception("Invalid input parameter %s." % key)
    
  # Update default conf with user values     
  defaults.__dict__.update(inp.__dict__)   
  
  # Finally, update conf                                
  inp.__dict__.update(defaults.__dict__)

  # Check some other stuff
  if len(inp.kbounds) != len(inp.kernel.pars):
    raise Exception("Input option ``kbounds`` has the wrong size!")
  if len(inp.kinit) != len(inp.kernel.pars):
    raise Exception("Input option ``kinit`` has the wrong size!")
  
  return inp

def GitHash():
  '''
  Get current git commit hash
  
  '''
  
  try:
    wrktree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gitpath = os.path.join(wrktree, '.git')
    GITHASH = subprocess.check_output(['git', '--git-dir=%s' % gitpath, 
                                       '--work-tree=%s' % wrktree, 'rev-parse', 'HEAD'], 
                                       stderr = subprocess.STDOUT)
    if GITHASH.endswith('\n'): GITHASH = GITHASH[:-1]
  except:
    GITHASH = ""
  
  return GITHASH

def Bold(string):
  '''
  Make a string bold-faced for Linux terminal output.
  
  '''
  
  return ("\x1b[01m%s\x1b[39;49;00m" % string)

def Help(param = None):
  '''
  
  '''
  
  helpstr = '\x1b[1mOptions:\x1b[0m ' + ', '.join(sorted([k for k in defaults.__dict__.keys() if not k.startswith('_')]))
  
  if param is not None:
    try:
      docstring = defaults._Docs().__dict__[param]
      print("\x1b[1m%s:\x1b[0m %s. \x1b[1mDefault:\x1b[0m %s" % (param, docstring, str(defaults.__dict__[param])))
    except:
      print("\x1b[1m%s:\x1b[0m No docstring available." % param)
  else:
    print(helpstr)

def GetOutliers(data, sig_tol = 3.):
  """
  Outlier rejection: see the
  `Wikipedia article <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_ .
  
  """
  
  M = np.nanmedian(data)
  MAD = 1.4826*np.nanmedian(np.abs(data - M))
  
  out = []
  for i, x in enumerate(data):
    if (x > M + sig_tol * MAD) or (x < M - sig_tol * MAD):
      out.append(i)    
  out = np.array(out, dtype = int)
  
  return out, M, MAD

def FreedmanDiaconis(x):
  '''
  Returns the optimal number of bins for a histogram
  of the vector quantity x according to the Freedman-
  Diaconis rule.
  
  '''
  q75, q25 = np.percentile(x, [75, 25])
  iqr = q75 - q25
  h = 2*iqr*len(x)**(-1./3.)
  bins = int((np.nanmax(x) - np.nanmin(x))/h)
  return bins

def GetData(id, data_type = 'prc', datadir = ''):
  '''
  
  '''

  try:
    data = np.load(os.path.join(datadir, str(id), '_data', '%s.npz' % data_type))['data'][()]
  except FileNotFoundError:
    raise FileNotFoundError("File ``_data/%s.npz`` not found." % data_type)

  return data

def SampleKOIs(datadir, N, exclude = ['FALSE POSITIVE']):
  '''
  
  '''
  
  # Disposition exclusions
  exclude = np.atleast1d(exclude)
  
  # Grab all KOIs
  client = kplr.API()
  kois = client.kois()

  # Find the indices of those we've processed already
  done = ["K%08.2f" % float(k) for k in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, k))]
  done = [i for i, k in enumerate(kois) if (k.kepoi_name in done)]

  # Now find the indices of those in multiplanet systems
  multi = []
  for i, koi in enumerate(kois):
    star, planet = koi.kepoi_name.split('.')
    if any([(k.kepoi_name.split('.')[0] == star and k.kepoi_name.split('.')[1] != planet) for k in kois]):
      multi.append(i)

  # Exclude certain dispositions
  exc = []
  for i, koi in enumerate(kois):
    if (koi.koi_disposition in exclude) or (koi.koi_pdisposition in exclude):
      exc.append(i)

  # Remove them!
  kois = [i for j, i in enumerate(kois) if j not in done + multi + exc]

  # Now return the numbers of ``N`` randomly chosen kois
  if N < len(kois):
    sample = [float(koi.kepoi_name[1:]) for koi in random.sample(kois, N)]
  else:
    sample = [float(koi.kepoi_name[1:]) for koi in kois]
    
  return sample

def Chunks(l, n):
  """
  Yield successive ``n``-sized chunks from ``l``. Merges the
  last two chunks if the last chunk would be smaller
  than ``n``.
  
  """
  
  for i in range(0, len(l), n):
    if i + 2 * n <= len(l):
      yield l[i:i+n]
    else:
      yield l[i:]
      break