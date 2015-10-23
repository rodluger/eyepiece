#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
lnlike.py
---------

>>> python launcher.py /path/to/input/script.py

'''

import subprocess
from eyepiece.inspect import Inspect
from eyepiece.download import GetData
from eyepiece.utils import Input
import os
import sys

if __name__ == '__main__':
  
  input_file = os.path.abspath(str(sys.argv[1]))
  inp = Input(input_file)
  
  # Try to load the data
  try:
    GetData(koi = inp.koi, data_type = 'bkg', datadir = inp.datadir)
  except:
    Inspect(input_file)

  subprocess.call(['qsub', '-vINPUTFILE=%s' % input_file, 'mpi.pbs'])