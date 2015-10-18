#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from eyepiece import Inspect, GetData

# Try to load the data
try:
  GetData(koi = 17.01, data_type = 'bkg')
except:
  Inspect(koi = 17.01, blind = True)

subprocess.call(['qsub', 'mpi.pbs'])