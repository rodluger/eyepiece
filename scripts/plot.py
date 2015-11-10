#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
plot.py
-------

Re-plots the detrending plots for all detrended KOIs.

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import matplotlib as mpl; mpl.use('Agg')
from eyepiece import Eyepiece
from eyepiece.utils import Input
import os

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError


inp = Input('input.py')
kois = [float(f) for f in os.listdir(inp.datadir) if '.' in f and os.path.isdir(os.path.join(inp.datadir, f))]

for koi in kois:

  print("Plotting %.2f..." % koi)

  # Copy the input file and add the appropriate ID
  with open('input.py', 'r') as f:
    contents = f.read()
  contents += '\n\n# Added by ``rand.py`` (overrides any ``id`` entries above)\nid = %.2f\n' % koi
    
  # Create a new input file
  tmp = 'tmp.py'
  with open(tmp, 'w') as f:
    print(contents, file = f)
  
  # Remove plots
  for file in ['detrended.png', 'folded.png', 'comparison.png']:
    try:
      os.remove(os.path.join(inp.datadir, str(koi), '_plots', file))
    except (FileNotFoundError, OSError):
      pass
  
  # Re-plot
  eye = Eyepiece('tmp.py')
  eye.Plot()
  
  # Remove input file
  os.remove('tmp.py')