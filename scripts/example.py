#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

Runs full detrending on a given KOI. Edit the file ``input.py``
in this directory for custom options.

'''

from eyepiece import Detrend, Plot
try:
  import para
  pool = para.Pool()
except:
  pool = None

if __name__ == '__main__':
      
  # Run detrending. You can optionally specify a 
  # parallelization pool instance here.
  #success = Detrend(input_file = 'input.py', pool = None)
  
  # DEBUG
  success = True
  
  # Plot the results
  if success:
    Plot(input_file = 'input.py')