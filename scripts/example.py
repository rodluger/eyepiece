#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example.py
----------

'''

import eyepiece
eyepiece.Inspect(koi = 254.01)
eyepiece.pld.PLD(koi = 254.01, quarters = range(1,18), niter = 2, maxfun = 10, debug = True)