#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test3.py
--------

A test of our decorrelation technique on quarter 4 of KOI 17.01.
   
'''

import matplotlib.pyplot as pl
import george
import numpy as np
from eyepiece.download import GetData
from scipy.optimize import fmin_l_bfgs_b

kernel = 1. * george.kernels.Matern32Kernel(1.) * george.kernels.Matern32Kernel(1.)
nkpars = 3
pld = False
kbounds = np.array([[1.e-8, 1.e8], [1.e-4, 1.e8], [1.e-4, 1.e8]])
kinit = np.array([10., 1., 10.])
data = None

def NegLnLike(coeffs):
  '''
  Returns the negative log-likelihood and its gradient for the model with coefficients ``coeffs``
  
  '''

  # Load the data (if necessary)
  global data, kernel, pld
  if data is None:
    data = GetData(17.01, data_type = 'bkg')
  quarter = data[4]

  # Pixel model coefficients
  if pld:
    d = coeffs[nkpars:]
  
  # Gaussian process coefficients
  kernel.pars = coeffs[:nkpars]
  
  # Our quasi-periodic kernel
  gp = george.GP(kernel = kernel)
  
  # The log-likelihood
  ll = 0
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):
  
    if pld:
      # The pixel model
      pmod = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)
    
      # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
      T = np.ones_like(fsum)
      ferr = np.zeros_like(fsum)
      for k, _ in enumerate(ferr):
        ferr[k] = np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)
    
    else:
      
      ferr = np.sqrt(np.sum(perr ** 2, axis = 1))
      pmod = np.ones_like(fsum) * np.median(fsum)
    
    # Evaluate the model
    try:      
    
      # Compute the likelihood
      gp.compute(time, ferr)
      ll += gp.lnlikelihood(fsum - pmod)

    except Exception as e:
        
      ll = -1.e10
      grad_ll = np.zeros_like(coeffs, dtype = float)
      break

  # DEBUG
  print(ll)

  return -ll
    
def Decorrelate():
  '''
  
  '''
  
  # Load the data (if necessary)
  global data, kinit, kbounds, pld
  if data is None:
    data = GetData(17.01, data_type = 'bkg')
  quarter = data[4]
  
  if pld:
    # Number of pixels in aperture
    npix = quarter['fpix'][0].shape[1]

    # Initial guess
    init = np.append(kinit, [np.median(quarter['fsum'][0])] * npix)
  
    # Bounds
    bounds = np.concatenate([kbounds, [[-np.inf, np.inf]] * npix])
  
  else:
    init = kinit
    bounds = kbounds

  # Run the optimizer  
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = True,
                      bounds = bounds,
                      m = 10, factr = 1.e1, epsilon = 1e-8,
                      pgtol = 1e-05)
  
  
  coeffs = res[0]
  lnlike = -res[1]
  info = res[2]       
  
  # Pixel coefficients 
  if pld: 
    d = coeffs[nkpars:]
  
  # Gaussian process coefficients
  kernel.pars = coeffs[:nkpars]
  
  gp = george.GP(kernel = kernel)
  
  all_time = []
  all_fsum = []
  all_pmod = []
  all_gpmu = []
  all_cerr = []
  all_yerr = []
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):

    if pld:
      # The PLD pixel model
      pm = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)
    
      # Correct error propagation
      T = np.ones_like(fsum)
      cerr = np.zeros_like(fsum)
      for k in range(len(fsum)):
        cerr[k] = np.sum(((1. / T[k]) + (pm[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)
    
    else:
    
      cerr = np.sqrt(np.sum(perr ** 2, axis = 1))
      pm = np.ones_like(fsum) * np.median(fsum)
    
    # Detrend with GP
    gp.compute(time, cerr)
    mu, cov = gp.predict(fsum - pm, time)
    yerr = np.sqrt(np.diag(cov))

    all_time.extend(time)
    all_fsum.extend(fsum)
    all_pmod.extend(pm)
    all_gpmu.extend(mu)
    all_cerr.extend(cerr)
    all_yerr.extend(yerr)
    
  time = np.array(all_time)
  fsum = np.array(all_fsum)
  pmod = np.array(all_pmod)
  gpmu = np.array(all_gpmu)
  cerr = np.array(all_cerr)  
  yerr = np.array(all_yerr)
  
  return {'time': time, 'fsum': fsum, 'pmod': pmod, 'gpmu': gpmu, 'cerr': cerr, 'yerr': yerr, 'coeffs': coeffs, 'lnlike': lnlike, 'info': info, 'init': init}


res = Decorrelate()
y = res['fsum'] - res['pmod'] - res['gpmu']
t = res['time']
yerr = res['yerr']

pl.plot(t, y, 'b.', alpha = 0.3)
pl.fill_between(t, y - yerr, y + yerr, alpha = 0.1, lw = 0, color = 'r')
pl.show()