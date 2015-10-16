#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test1.py
--------

A test routine to compare the numerical and analytical gradients of our
log-likelihood function. I suspect my formula for the derivative of the
log-likelihood with respect to the pixel coefficients is wrong, since I
assume the covariance matrix is not a function of the pixel coefficients.
This is not actually true, since the PLD coefficients affect the errors
of the data and make their way into the main diagonal of this matrix.
I need to talk to Eric about how to do the error propagation correctly...
   
'''

from eyepiece.download import GetData
import george
import numpy as np; np.random.seed(1234)

def Compute(coeffs, time, fsum, fpix, perr):

  # Pixel model coefficients
  d = coeffs[3:]

  # Gaussian process coefficients
  a = coeffs[0]
  b = coeffs[1]
  c = coeffs[2]

  # Our quasi-periodic kernel
  gp = george.GP(kernel = a * george.kernels.Matern32Kernel(b) * george.kernels.CosineKernel(c))
     
  # The pixel model
  pmod = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)

  # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
  T = np.ones_like(fsum)
  ferr = np.zeros_like(fsum)
  for k, _ in enumerate(ferr):
    ferr[k] = np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)

  # Compute the likelihood
  gp.compute(time, ferr)
  
  return gp, pmod
  
def AnalyticalGradient(coeffs, time, fsum, fpix, perr):

  # Compute the gradient of the likelihood with some badass linear algebra   
  gp, pmod = Compute(coeffs, time, fsum, fpix, perr)
  A = fpix.T / fsum
  grad_ll_pld = -np.dot(A, gp.solver.apply_inverse(fsum - pmod))    
  grad_ll = np.append(gp.grad_lnlikelihood(fsum - pmod) / coeffs[:3], grad_ll_pld)
  
  return grad_ll

def NumericalGradient(coeffs, time, fsum, fpix, perr, dx = 1.e-12):

  # Compute it numerically
  gp, pmod = Compute(coeffs, time, fsum, fpix, perr)
  ll0 = gp.lnlikelihood(fsum - pmod)
  res = np.zeros_like(coeffs)
  
  for i, c in enumerate(coeffs):
    cprime = np.array(coeffs)
    cprime[i] *= (1. + dx)
    
    gp, pmod = Compute(cprime, time, fsum, fpix, perr)
    ll = gp.lnlikelihood(fsum - pmod)
    
    res[i] = (ll - ll0)/(cprime[i] - coeffs[i])
  
  return res



# The quarter #
q = 6

# The chunk #
n = 0

# The data
data = GetData(17.01, data_type = 'bkg')
quarter = data[q]
npix = quarter['fpix'][n].shape[1]
coeffs = np.array([10., 10., 10.] + [np.median(quarter['fpix'][n])] * npix)
coeffs = coeffs * (1 + 0.25 * np.random.randn(len(coeffs)))

time = quarter['time'][n]
fsum = quarter['fsum'][n]
fpix = quarter['fpix'][n]
perr = quarter['perr'][n]


print(AnalyticalGradient(coeffs, time, fsum, fpix, perr))
print(NumericalGradient(coeffs, time, fsum, fpix, perr))


