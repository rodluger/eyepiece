#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
lnlike.py
---------

Runs full detrending on a given KOI.

>>> python launcher.py [/path/to/input/script.py]

'''

import subprocess
from eyepiece.inspect import Inspect
from eyepiece.download import GetData
from eyepiece.utils import Input
import os
import sys

if __name__ == '__main__':
  
  # Did the user specify an input file?
  if len(sys.argv) == 2:
    input_file = os.path.abspath(str(sys.argv[1]))
  else:
    # Assume it's in the cwd
    input_file = 'input.py'
  # Let's try to load it
  try:
    inp = Input(input_file)
  except:
    raise Exception("Please provide a valid input file!")
  
  # Try to load the data
  try:
    GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  except IOError:
    success = Inspect(input_file)
    if not success:
      if not inp.quiet:
        print("Detrending aborted!")
      sys.exit()

  # Set options
  vars = '-vINPUTFILE=%s' % input_file
  resources = '-l nodes=%(N)d:ppn=%(PPN)d,feature=%(PPN)dcore,mem=%(M)dgb,walltime=%(W)d:00:00' % \
              {'N': inp.nodes, 'PPN': inp.ppn, 'M': inp.nodes * inp.mpn, 'W': inp.walltime}
  
  # QSUB
  subprocess.call(['qsub', vars, resources, 'mpi.pbs'])