#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pld.py
------

An example of our PLD + GP decorrelation method for KOI 254.01.

'''

from eyepiece import GetData
from eyepiece.config import dir
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as pl
import george
import os
import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  from statsmodels.tsa.stattools import acf
import scipy.signal as signal

def NegLnLike(coeffs, quarter, return_gradient = True, debug = False):
  '''
  Returns the negative log-likelihood and its gradient for the model with coefficients ``coeffs``
  
  '''

  # Pixel model coefficients
  d = coeffs[3:]
  
  # Gaussian process coefficients
  a = coeffs[0]
  b = coeffs[1]
  c = coeffs[2]
  
  # Our quasi-periodic kernel
  gp = george.GP(kernel = (a ** 2) * george.kernels.Matern32Kernel(b ** 2) * george.kernels.CosineKernel(c))
  
  # The log-likelihood
  ll = 0
  
  # Its gradient
  grad_ll = np.zeros_like(coeffs)
  
  for chunk in quarter:
  
    # The flux data for this chunk
    fpix = chunk.mask.fpix
    fsum = chunk.mask.fsum
    perr = chunk.mask.perr
  
    # The pixel model
    pmod = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)
    
    # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
    T = np.ones_like(fsum)
    ferr = np.zeros_like(fsum)
    for k, _ in enumerate(ferr):
      ferr[k] = np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)
    
    # Evaluate the model
    try:
      gp.compute(chunk.mask.time, ferr)
      ll += gp.lnlikelihood(chunk.mask.fsum - pmod)
    except:
      if return_gradient:
        return 1.e10, np.zeros_like(coeffs, dtype = float)
      else:
        return 1.e10

    # Compute the gradient of the likelihood with some badass linear algebra   
    A = fpix.T / fsum
    grad_ll_pld = -np.dot(A, gp.solver.apply_inverse(fsum - pmod))    
    grad_ll += np.append(gp.grad_lnlikelihood(fsum - pmod), grad_ll_pld)

  # Print the likelihood and the gradient?
  if debug:
    print(ll, grad_ll)

  if return_gradient:
    return -ll, -grad_ll
  else:
    return -ll

def Decorrelate(quarter, maxfun = 15000, approx_grad = False, debug = False):
  '''
  
  '''
   
  # Number of pixels in aperture       
  npix = quarter[0].mask.fpix.shape[1]
  
  # Below we construct our initial guess:
  
  # Let's find the analytical solution, c_j = (A_jm)^-1 * B_m
  fpix = np.array([x for chunk in quarter for x in chunk.mask.fpix])
  fsum = np.array([x for chunk in quarter for x in chunk.mask.fsum])
  time = np.array([x for chunk in quarter for x in chunk.mask.time])
  A = np.zeros((npix, npix))
  for j in range(npix):
    for m in range(npix):
    
      # This could be sped up!
      A[j][m] = np.sum( fpix[:, j] * fpix[:, m] / fsum ** 2 , axis = 0)
      
  B = np.sum(fpix, axis=0)  
  cj = np.dot(np.linalg.inv(A), B)
  y = fsum - np.sum(fpix * np.outer(1. / fsum, cj), axis = 1)
  
  # Timescale (1 <= tau <= 20)
  acor = acf(y, nlags = len(y))[1:]
  t = np.linspace(0, time[-1] - time[0], len(y) - 1)
  tau = min(max(1., t[np.argmax(y < 0)]), 20.)
  
  # Amplitude (standard deviation of PLD-decorrelated data)
  amp = np.std(y)
  
  # Period (1 <= per <= 100)
  par = np.linspace(100, 1, 101)
  pdg = signal.lombscargle(time, y, 2. * np.pi / par)
  per = par[np.argmax(pdg)]
  
  # Initial guess
  init = np.append([amp, tau, per], cj)

  # Very loose physical bounds
  bounds = [[0, 1.e4], [0, 1.e4], [0, 1.e4]] + [[-np.inf, np.inf]] * npix
  
  # Run the optimizer
  args = [quarter, not approx_grad, debug]
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = approx_grad,
                      args = args, bounds = bounds,
                      m = 10, factr = 1.e1, epsilon = 1e-8,
                      pgtol = 1e-05, maxfun = maxfun)
  coeffs = res[0]
  lnlike = -res[1]
  info = res[2]       
  a = res[0][0]
  b = res[0][1]
  c = res[0][2]
  d = res[0][3:]
  gp = george.GP(kernel = (a ** 2) * george.kernels.Matern32Kernel(b ** 2) * george.kernels.CosineKernel(c))
  
  time = []
  fsum = []
  pmod = []
  gpmu = []
  yerr = []
  
  for chunk in quarter:
  
    # The PLD pixel model
    pm = np.sum(chunk.mask.fpix * np.outer(1. / chunk.mask.fsum, d), axis = 1)
    
    # Correct error propagation
    T = np.ones_like(chunk.mask.fsum)
    cerr = np.zeros_like(chunk.mask.fsum)
    for k in range(len(chunk.mask.ferr)):
      cerr[k] = np.sum(((1. / T[k]) + (pm[k] / chunk.mask.fsum[k]) - (d / chunk.mask.fsum[k])) ** 2 * chunk.mask.perr[k] ** 2)
    
    # Detrend with GP
    gp.compute(chunk.mask.time, cerr)
    mu, cov = gp.predict(chunk.mask.fsum - pm, chunk.mask.time)

    time.extend(chunk.mask.time)
    fsum.extend(chunk.mask.fsum)
    pmod.extend(pm)
    gpmu.extend(mu)
    yerr.extend(cerr)
    
  time = np.array(time)
  fsum = np.array(fsum)
  pmod = np.array(pmod)
  gpmu = np.array(gpmu)
  yerr = np.array(yerr)  
  
  return time, fsum, pmod, gpmu, yerr, coeffs, lnlike, info
  
def Run(koi = 254.01, quarters = range(18), debug = False):
  '''
  
  '''
  
  # Load the raw data
  data = GetData(koi)
  
  # Set up a function for the mapping
  def worker(q):
    quarter = data[q]
    if quarter['time'] == []: 
      return False
    if not os.path.exists(os.path.join(dir, str(koi), 'pld')):
      os.makedirs(os.path.join(dir, str(koi), 'pld'))
    np.savez(os.path.join(dir, str(koi), 'pld', '%02d.npz' % q), 
             data = Decorrelate(quarter, debug = debug))
    return True

  # Decorrelate the data for each quarter
  map(worker, quarters)

def Plot(koi = 254.01, quarters = range(18)):
  '''
  
  '''
  
  # Plot the decorrelated data
  fig, ax = pl.subplots(3, 1, figsize = (48, 16)) 
  lt = [None] * (quarters[-1] + 1)
  wf = [""] * (quarters[-1] + 1)
  fc = np.zeros(quarters[-1] + 1)
  ni = np.zeros(quarters[-1] + 1)
  ll = np.zeros(quarters[-1] + 1)
  cc = [None] * (quarters[-1] + 1)
  for q in quarters:
    
    # Load the decorrelated data
    try:
      time, fsum, pmod, gpmu, yerr, coeffs, lnlike, info = np.load(os.path.join(dir, str(koi), 'pld', '%02d.npz' % q))['data']
    except IOError:
      continue
    
    # The SAP flux
    ax[0].plot(time, fsum, 'k.', alpha = 0.3)
    
    # The PLD-detrended SAP flux (blue) and the GP (red)
    ax[1].plot(time, fsum - pmod, 'b.', alpha = 0.3)
    ax[1].plot(time, gpmu, 'r-')
  
    # The fully detrended flux
    y = fsum - pmod - gpmu
    ax[2].plot(time, y, 'b.', alpha = 0.3)
    ax[2].fill_between(time, y - yerr, y + yerr, alpha = 0.1, lw = 0, color = 'r')
    
    # Appearance
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[0].margins(0, 0.01)
    ax[1].margins(0, 0.01)
    ax[2].margins(0, 0.01)
    
    # Extra info
    lt[q] = time[-1]
    fc[q] = info['funcalls']
    ni[q] = info['nit']
    if info['warnflag']:
      wf[q] = info['task']
      if len(wf[q]) > 20:
        wf[q] = "%s..." % wf[q][:20]
    ll[q] = lnlike
    cc[q] = coeffs
      
  ltq = ax[0].get_xlim()[0]  
  yp0 = ax[0].get_ylim()[1]
  yp2 = ax[2].get_ylim()[1]
  for q in quarters:  
    if lt[q] is None:
      continue
    ax[0].annotate(q, ((ltq + lt[q]) / 2., yp0), ha='center', va='bottom', fontsize = 24)
    if wf[q] == "":
      ax[2].annotate("\n   SUCCESS", (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'b')
    else:
      ax[2].annotate("\n   ERROR: %s" % wf[q], (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'r')
    ax[2].annotate("\n\n   CALLS: %d" % fc[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
    ax[2].annotate("\n\n\n   NITR: %d" % ni[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
    ax[2].annotate("\n\n\n\n   LNLK: %.2f" % ll[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
    for axis in ax:
      axis.axvline(lt[q], color='k', ls = '--')
    ltq = lt[q]
    
  fig.savefig(os.path.join(dir, str(koi), 'pld', 'decorr.png'), bbox_inches = 'tight')

if __name__ == '__main__':
    
  calc = True
  debug = False
  koi = 254.01
  
  if calc:
    Run(koi = koi, debug = debug)
  
  Plot(koi = koi)