#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
transits.py
-----------


'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .download import GetData

__all__ = ['ViewTransits']

def ViewTransits(koi = 17.01, blind = False):
  GetData(koi, data_type = 'trn', blind = blind)
