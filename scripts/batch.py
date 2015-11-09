#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
batch.py
--------

Runs ``eyepiece`` in batch mode.

'''

from eyepiece import Eyepiece
import subprocess
import os

for file in os.listdir('batch'):
  if file.endswith('.py'):
    eye = Eyepiece('batch/%s' % file)
    subprocess.call(['mpi', 'hyak.py', '-i', 'batch/%s' % file])