#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
rand.py
-------

Runs ``eyepiece`` on a random sample of single-planet KOIs,
grabbing input params from ``input.py``.

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from eyepiece import Eyepiece
from eyepiece.utils import SampleKOIs, Input
import subprocess
import os
import sys
import shutil

# Let's do 10 KOIs, unless the user specifies a number
try:
  N = int(sys.argv[1])
except:
  N = 10

# Grab the input file
inp = Input('input.py')

# Delete and recreate ``random`` dir
if os.path.exists('random'):
  shutil.rmtree('random')
os.makedirs('random')

# Loop over each one

print('Generating KOI sample...')

for i, koi in enumerate(SampleKOIs(inp.datadir, N)):

  print('Running KOI %.2f...' % koi)
  
  # Copy the input file and add the appropriate ID
  with open('input.py', 'r') as f:
    contents = f.read()
  contents += '\n\n# Added by ``rand.py`` (overrides any ``id`` entries above)\nid = %.2f' % koi
      
  # Create a new input file
  tmp = 'random/input%04d.py' % i
  with open(tmp, 'w') as f:
    print(contents, file = f)
  
  # Run eyepiece
  eye = Eyepiece(tmp)
  subprocess.call(['mpi', 'hyak.py', '-i', tmp, 
                   '-a', tmp, '-l', '%s.log' % tmp])