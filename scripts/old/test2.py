#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test2.py
--------

Same as ``test1.py``, but there are no GPs here.
   
'''

from eyepiece.download import GetData
import numpy as np; np.random.seed(1234)

def Compute(coeffs, time, fsum, fpix, perr):

  pmod = np.sum(fpix * np.outer(1. / fsum, coeffs), axis = 1)
  
  # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
  T = np.ones_like(fsum)
  ferr = np.zeros_like(fsum)
  for k, _ in enumerate(ferr):
    ferr[k] = np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (coeffs / fsum[k])) ** 2 * perr[k] ** 2)

  # debug
  #ferr = np.ones_like(pmod) * 57
  
  return pmod, ferr
  
def AnalyticalGradient(coeffs, time, fsum, fpix, perr):

  pmod, ferr = Compute(coeffs, time, fsum, fpix, perr)
  A = fpix.T / fsum
  
  CINV = np.eye(len(ferr)) * 1. / (ferr ** 2)
  
  grad_ll = np.dot(A, np.dot(CINV, (fsum - pmod)))
  
  return grad_ll

def NumericalGradient(coeffs, time, fsum, fpix, perr, dx = 1.e-6):

  # Compute it numerically
  pmod0, ferr0 = Compute(coeffs, time, fsum, fpix, perr)
  ll0 = -0.5 * np.sum( ((fsum - pmod0) / ferr0) ** 2 )
  
  res = np.zeros_like(coeffs)
  for i, c in enumerate(coeffs):
    cprime = np.array(coeffs)
    cprime[i] *= (1. + dx)
    
    pmod, ferr = Compute(cprime, time, fsum, fpix, perr)
    ll = -0.5 * np.sum( ((fsum - pmod) / ferr) ** 2 )
    
    res[i] = (ll - ll0)/(cprime[i] - coeffs[i])
    
  return res



# The quarter
q = 6
# The chunk
n = 0
# The data
data = GetData(17.01, data_type = 'bkg')
quarter = data[q]
npix = quarter['fpix'][n].shape[1]
coeffs = np.array([np.median(quarter['fpix'][n])] * npix)
coeffs = coeffs * (1 + 0.25 * np.random.randn(len(coeffs)))
time = quarter['time'][n]
fsum = quarter['fsum'][n]
fpix = quarter['fpix'][n]
perr = quarter['perr'][n]


print(AnalyticalGradient(coeffs, time, fsum, fpix, perr))
print(NumericalGradient(coeffs, time, fsum, fpix, perr))


