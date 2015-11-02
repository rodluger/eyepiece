#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
utils.py
--------

'''

from . import defaults
import os
import subprocess
import imp
import numpy as np
import types

__all__ = ['Input', 'GitHash', 'Bold', 'RowCol', 'FunctionWrapper']

def Input(input_file = None):
  '''
  
  '''
  
  if input_file is None:
    return defaults
  
  try:
    inp = imp.load_source("input", input_file) 
  except:
    raise IOError("Please provide a valid input file!")

  # Let's check that the user didn't specify both a KIC and a KOI number
  if ('kic' in inp.__dict__.keys()) and ('koi' in inp.__dict__.keys()):
    if (inp.kic is not None) and (inp.koi is not None):
      raise Exception("Please specify either a KOI or a KIC id, but not both!")

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

  # Set the id
  if inp.kic is not None:
    inp.koi = None
    inp.id = inp.kic
  elif inp.koi is not None:
    inp.id = inp.koi

  # Check some other stuff
  if inp.order is not None:
    inp.kernel = None
    if len(inp.kbounds) != inp.order + 1:
      raise Exception("Input option ``kbounds`` has the wrong size!")
    if len(inp.kinit) != inp.order + 1:
      raise Exception("Input option ``kinit`` has the wrong size!")
  elif inp.kernel is not None:
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

def RowCol(N):
  '''
  Given an integer ``N``, returns the ideal number of columns and rows 
  to arrange ``N`` subplots on a grid.
  
  :param int N: The number of subplots
  
  :returns: **``(cols, rows)``**, the most aesthetically pleasing number of columns \
  and rows needed to display ``N`` subplots on a grid.
  
  '''
  rows = np.floor(np.sqrt(N)).astype(int)
  while(N % rows != 0):
    rows = rows - 1
  cols = N/rows
  while cols/rows > 2:
    cols = np.ceil(cols/2.).astype(int)
    rows *= 2
  if cols > rows:
    tmp = cols
    cols = rows
    rows = tmp
  return cols, rows

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

class FunctionWrapper(object):
  '''
  
  '''
  def __init__(self, f, *args, **kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, x):
    return self.f(x, *self.args, **self.kwargs)