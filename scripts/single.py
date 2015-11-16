#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
single.py
---------

Runs ``eyepiece`` on a single system described in ``input.py``

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from eyepiece import Eyepiece
import subprocess
import os

eye = Eyepiece('input.py')
subprocess.call(['mpi', 'hyak.py', '-i', 'input.py', 
                 '-a', 'input.py', '-l', 'input.py.log'])