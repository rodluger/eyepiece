#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
utils.py
--------

'''

import os
import subprocess

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
