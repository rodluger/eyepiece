#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
lnlike.py
---------

>>> python launcher.py -i /path/to/input/script.py

'''

import subprocess
from eyepiece.inspect import Inspect
from eyepiece.download import GetData
from eyepiece.utils import Input
import argparse
import os

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(prog = 'launcher')
  parser.add_argument("-i", "--input", default = None, help = 'Input file for this run')
  args = parser.parse_args()
  
  if args.input is not None:
    input_file = os.path.abspath(args.input)
    input = Input(input_file)
  else:
    input_file = "none"
    input = Input()
  
  # Try to load the data
  try:
    GetData(koi = input.koi, data_type = 'bkg', datadir = input.datadir)
  except:
    Inspect(input_file)

  subprocess.call(['qsub', '-vINPUT=%s' % input_file, 'mpi.pbs'])