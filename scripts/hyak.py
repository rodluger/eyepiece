#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
hyak.py
-------

Run ``eyepiece`` on Hyak.

>>> python hyak.py

'''

from eyepiece import Eyepiece
import sys
import para

if __name__ == '__main__':

  # Grab the data
  eye = Eyepiece('input.py')

  # If no arguments were provided, re-run this script with ``mpi`` to add the job
  # to the PBS queue
  if len(sys.argv) == 1:
    subprocess.call(['mpi', 'hyak.py', '-a', 'QSUB'])
  
  # We're already in a PBS run. Let's detrend!
  else:
  
    # Create a ``para`` MPI pool instance
    pool = para.Pool()
    eye.Detrend(pool = pool)
    pool.close()
    
    # Plot the results
    eye.Plot()