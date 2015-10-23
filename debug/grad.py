#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
grad.py
-------

A test of our analytical log-likelihood gradient calculation. It's
faster than the numerical gradient by a factor of ~2x in the example case:

>>> Numerical time:  1.5756903096000315
>>> Analytical time: 0.7447342678002315

For a smaller chunk (Quarter 4, chunk 1), it's even faster:

>>> Numerical time:  0.15853579879985774
>>> Analytical time: 0.046326947699708396

Note that for some reason ``SlowAnalyticalGradient`` give numbers that are in
better agreement with ``NumericalGradient``, even though the math is (or should be!)
identical. The disagreement is at the 1e-6 level, which is below our precision
anyways, but I should look into this eventually.

'''

from eyepiece import GetData
import numpy as np; np.random.seed(123)
import george
import timeit

def AnalyticalGradient(coeffs, time, fsum, fpix, perr, kernel = 1. * george.kernels.Matern32Kernel(1.), tmod = None, lndet = True):
  
  # PLD coefficients
  c = coeffs[len(kernel.pars):]
  
  # Kernel params
  kernel.pars = coeffs[:len(kernel.pars)]
  
  # Number of data points
  K = len(time)
  
  # Inverse of the transit model
  if tmod is None:
    D = np.ones_like(time)
  else:
    D = 1. / tmod
  
  # The pixel model
  pmod = np.sum(fpix * np.outer(1. / fsum, c), axis = 1)
  
  # The PLD-detrended data
  y = fsum - pmod

  # Errors on detrended data (y)
  X = D + pmod / fsum
  B = X.reshape(K, 1) * perr - c * perr / fsum.reshape(K, 1)
  yerr = np.sum(B ** 2, axis = 1) ** 0.5

  # Compute the likelihood
  gp = george.GP(kernel)
  gp.compute(time, yerr)
  ll = gp.lnlikelihood(y)

  # Compute the gradient of the likelihood with respect to the PLD coefficients
  # First, the gradient assuming the covariance is independent of the coeffs
  A = gp.solver.apply_inverse(y)
  R = fpix / fsum.reshape(K, 1)
  grad_pld = np.dot(R.T, A)
  
  # Pre-compute some stuff
  H = np.array([perr[k] * (X[k] - (c / fsum[k])) for k in range(K)])
  J = np.array([perr[k] / fsum[k] for k in range(K)])
  
  # Are we computing the very small contribution from the ln-det term?
  # If so, pre-compute the diagonal of the inverse covariance matrix
  if lndet:
    KID = np.diag(gp.solver.apply_inverse(np.eye(K)))
  
  # Now add the covariance term  
  for m, _ in enumerate(grad_pld):

    # The Kronecker delta
    KD = np.zeros_like(grad_pld); KD[m] = 1
    
    # Derivative of sigma^2 with respect to the mth PLD coefficient
    dS2dC = 2 * np.diag(np.sum(H * (J * R[:,m].reshape(K,1) - np.outer(J[:,m], KD)), axis = 1))
    
    # The covariance term
    grad_pld[m] += 0.5 * np.dot(A.T, np.dot(dS2dC, A))
    
    if lndet:
      # The small ln(det) term
      grad_pld[m] -= np.dot(KID, np.diag(dS2dC))
  
  grad_ll = np.append(gp.grad_lnlikelihood(fsum - pmod) / kernel.pars, grad_pld)
  
  return grad_ll  
  
def SlowAnalyticalGradient(coeffs, time, fsum, fpix, perr):
  
  # The pixel model
  pmod = np.sum(fpix * np.outer(1. / fsum, coeffs[2:]), axis = 1)

  # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
  T = np.ones_like(fsum)
  ferr = np.zeros_like(fsum)
  for k, _ in enumerate(ferr):
    ferr[k] = np.sqrt(np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (coeffs[2:] / fsum[k])) ** 2 * perr[k] ** 2))

  # Compute the likelihood
  kernel = 1. * george.kernels.Matern32Kernel(1.)
  kernel.pars = coeffs[:2]
  gp = george.GP(kernel)
  gp.compute(time, ferr)
  ll = gp.lnlikelihood(fsum - pmod)

  # Compute the gradient
  grad_pld = np.zeros_like(coeffs[2:])
  A = gp.solver.apply_inverse(fsum - pmod)
  N = len(time)
  for m, _ in enumerate(grad_pld):

    KD = np.zeros_like(grad_pld)
    KD[m] = 1

    dYdC = - fpix[:,m] / fsum
    
    dSdC = np.zeros((N,N))
    for n in range(N):
      dSdC[n,n] = 2 * np.sum(perr[n] * (1. / T[n] + pmod[n] / fsum[n] - (coeffs[2:] / fsum[n])) * (perr[n] * fpix[n,m] / fsum[n] ** 2 - KD * perr[n,m] / fsum[n] ))
  
    B = gp.solver.apply_inverse(dSdC)
    
    grad_pld[m] = -0.5 * ( 2 * np.dot(dYdC.T, A) - np.dot(A.T, np.dot(dSdC, A)) + np.trace(B))
  
  grad_ll = np.append(gp.grad_lnlikelihood(fsum - pmod) / kernel.pars, grad_pld)
  
  return grad_ll

def NumericalGradient(coeffs, time, fsum, fpix, perr, eps = 1.e-6):
  
  grad_ll = np.zeros_like(coeffs)
  
  # The log-likelihood
  pmod = np.sum(fpix * np.outer(1. / fsum, coeffs[2:]), axis = 1)
  T = np.ones_like(fsum)
  ferr = np.zeros_like(fsum)
  for k, _ in enumerate(ferr):
    ferr[k] = np.sqrt(np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (coeffs[2:] / fsum[k])) ** 2 * perr[k] ** 2))
  kernel = 1. * george.kernels.Matern32Kernel(1.)
  kernel.pars = coeffs[:2]
  gp = george.GP(kernel)
  gp.compute(time, ferr)
  ll = gp.lnlikelihood(fsum - pmod)
  
  # Perturb to compute the gradient
  for i, c in enumerate(coeffs):
    
    cprime = np.array(coeffs)
    cprime[i] = coeffs[i] * (1 + eps)
  
    # The pixel model
    pmod = np.sum(fpix * np.outer(1. / fsum, cprime[2:]), axis = 1)

    # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
    T = np.ones_like(fsum)
    ferr = np.zeros_like(fsum)
    for k, _ in enumerate(ferr):
      ferr[k] = np.sqrt(np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (cprime[2:] / fsum[k])) ** 2 * perr[k] ** 2))

    # Compute the likelihood
    kernel = 1. * george.kernels.Matern32Kernel(1.)
    kernel.pars = cprime[:2]
    gp = george.GP(kernel)
    gp.compute(time, ferr)
    grad_ll[i] = (gp.lnlikelihood(fsum - pmod) - ll) / (cprime[i] - coeffs[i])

  return grad_ll

if __name__ == '__main__':
  '''
  Run a timing test on quarter 4 of KOI 17.01.
  
  '''
  
  koi = 17.01
  n = 2
  q = 1
  data = GetData(koi, data_type = 'bkg')
  quarter = data[q]
  time = quarter['time'][n]
  fpix = quarter['fpix'][n]
  fsum = quarter['fsum'][n]
  perr = quarter['perr'][n]
  npix = fpix.shape[1]
  coeffs = np.append(np.abs(np.random.randn(2) * 10), np.random.randn(npix) * fsum[0])

  n = lambda: NumericalGradient(coeffs, time, fsum, fpix, perr)
  a = lambda: AnalyticalGradient(coeffs, time, fsum, fpix, perr)

  print('Numerical time: ', timeit.timeit(n, number = 10)/10)
  print('Analytical time:', timeit.timeit(a, number = 10)/10)
