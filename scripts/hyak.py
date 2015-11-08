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
import subprocess
import para

if __name__ == '__main__':

  # If no arguments were provided, download the data, then re-run this script 
  # with ``mpi`` to add the job to the PBS queue
  if len(sys.argv) == 1:
  
    # Download the data
    eye = Eyepiece('input.py')
    
    # Submit the job
    subprocess.call(['mpi', 'hyak.py', '-a', 'QSUB'])
  
  # We're already in a PBS run. Let's load the data, then detrend!
  else:
  
    # Create a ``para`` MPI pool instance
    pool = para.Pool()
    eye = Eyepiece('input.py')
    eye.Detrend(pool = pool)
    eye.Compare(pool = pool)
    pool.close()
    
    # Plot the results
    eye.Plot()