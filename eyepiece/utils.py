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

__all__ = ['Input', 'GitHash', 'Bold', 'RowCol']

def Input(input_file = None):
  '''
  
  '''
  
  if input_file is None:
    return defaults
  
  input = imp.load_source("input", input_file)                                       

  for key, val in list(input.__dict__.items()):
    if key.startswith('_'):
      input.__dict__.pop(key, None)
    else:
      # Check if user provided something they shouldn't have
      if key not in defaults.__dict__.keys():                                  
        raise Exception("Invalid input parameter %s." % key)
    
  # Update default conf with user values     
  defaults.__dict__.update(input.__dict__)   
  
  # Finally, update conf                                
  input.__dict__.update(defaults.__dict__)

  return input

def GitHash():
  '''
  Get current git commit hash
  '''
  
  try:
    _wrktree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _gitpath = os.path.join(_wrktree, '.git')
    GITHASH = subprocess.check_output(['git', '--git-dir=%s' % _gitpath, 
                                       '--work-tree=%s' % _wrktree, 'rev-parse', 'HEAD'], 
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