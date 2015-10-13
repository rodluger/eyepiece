#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pld.py
------

An example of our PLD + GP decorrelation method for KOI 254.01.

.. todo::
   - We need a better initial guess for the coefficients.

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .download import GetData
from eyepiece.config import datadir
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as pl
import george
import os
import warnings
try:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.stattools import acf
except ImportError:
  acf = None
import scipy.signal as signal

__all__ = ['Run', 'Plot']
data = None

def NegLnLike(coeffs, koi, q, debug = False):
  '''
  Returns the negative log-likelihood and its gradient for the model with coefficients ``coeffs``
  
  '''

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  quarter = data[q]

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
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):
  
    # The pixel model
    pmod = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)
    
    # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
    T = np.ones_like(fsum)
    ferr = np.zeros_like(fsum)
    for k, _ in enumerate(ferr):
      ferr[k] = np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)
    
    # Evaluate the model
    try:      
    
      print("CHECKPOINT A")
      
      import pdb; pdb.set_trace()
    
      # Compute the likelihood
      gp.compute(time, ferr)
      
      print("CHECKPOINT B")
      
      ll += gp.lnlikelihood(fsum - pmod)
    
      print("CHECKPOINT C")
    
      # Compute the gradient of the likelihood with some badass linear algebra   
      A = fpix.T / fsum
      grad_ll_pld = -np.dot(A, gp.solver.apply_inverse(fsum - pmod))    
      grad_ll += np.append(gp.grad_lnlikelihood(fsum - pmod), grad_ll_pld)
    
      print("CHECKPOINT D")
    
    except Exception as e:
      
      # Return a low likelihood
      if debug:
        print("Exception evaluating the Ln-Like func:", str(e))
        
      ll = -1.e10
      grad_ll = np.zeros_like(coeffs, dtype = float)
      break

  # Print the likelihood and the gradient?
  if debug:
    print("Ln-Like: ", ll)
    print("Grad-LL: ", grad_ll)

  return (-ll, -grad_ll)
    
def Decorrelate(koi, q, init, maxfun = 15000, debug = False):
  '''
  
  '''
  
  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  quarter = data[q]
  
  # Number of pixels in aperture
  npix = quarter['fpix'][0].shape[1]

  # Very loose physical bounds
  bounds = np.array([[0, 1.e4], [0, 1.e4], [0, 1.e4]] + [[-np.inf, np.inf]] * npix)
  
  # Run the optimizer  
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = False,
                      args = [koi, q, debug], bounds = bounds,
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
  
  all_time = []
  all_fsum = []
  all_pmod = []
  all_gpmu = []
  all_yerr = []
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):

    # The PLD pixel model
    pm = np.sum(fpix * np.outer(1. / fsum, d), axis = 1)
    
    # Correct error propagation
    T = np.ones_like(fsum)
    cerr = np.zeros_like(fsum)
    for k in range(len(fsum)):
      cerr[k] = np.sum(((1. / T[k]) + (pm[k] / fsum[k]) - (d / fsum[k])) ** 2 * perr[k] ** 2)
    
    # Detrend with GP
    gp.compute(time, cerr)
    mu, cov = gp.predict(fsum - pm, time)

    all_time.extend(time)
    all_fsum.extend(fsum)
    all_pmod.extend(pm)
    all_gpmu.extend(mu)
    all_yerr.extend(cerr)
    
  time = np.array(all_time)
  fsum = np.array(all_fsum)
  pmod = np.array(all_pmod)
  gpmu = np.array(all_gpmu)
  yerr = np.array(all_yerr)  
  
  return {'time': time, 'fsum': fsum, 'pmod': pmod, 'gpmu': gpmu, 'yerr': yerr, 'coeffs': coeffs, 'lnlike': lnlike, 'info': info}

def InitialGuess(koi, q, seed = None, sigma = 0.1):
  '''
  
  '''
  
  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  quarter = data[q]
      
  # Number of pixels in aperture       
  npix = quarter['fpix'][0].shape[1]
    
  # Let's find the analytical solution, c_j = (A_jm)^-1 * B_m
  fpix = np.array([x for chunk in quarter['fpix'] for x in chunk])
  fsum = np.array([x for chunk in quarter['fsum'] for x in chunk])
  time = np.array([x for chunk in quarter['time'] for x in chunk])

  A = np.zeros((npix, npix))
  for j in range(npix):
    for m in range(npix):
      # TODO: This could be sped up!
      A[j][m] = np.sum( fpix[:, j] * fpix[:, m] / fsum ** 2 , axis = 0)
      
  B = np.sum(fpix, axis=0)  
  cj = np.dot(np.linalg.inv(A), B)
  y = fsum - np.sum(fpix * np.outer(1. / fsum, cj), axis = 1)
  
  # Timescale (1 <= tau <= 20)
  if acf is not None:
    acor = acf(y, nlags = len(y))[1:]
    t = np.linspace(0, time[-1] - time[0], len(y) - 1)
    tau = min(max(1., t[np.argmax(acor < 0)] / 2.), 20.)                              # ~ Half the time it takes for acor to drop to zero
  else:
    tau = 10.
  
  # Amplitude (standard deviation of PLD-decorrelated data)
  amp = np.std(y)
  
  # Period (1 <= per <= 100)
  par = np.linspace(100, 1, 101)
  pdg = signal.lombscargle(time, y, 2. * np.pi / par)
  per = par[np.argmax(pdg)]
  
  # Initial guess
  init = np.append([amp, tau, per], cj)
  
  # Perturb it by sigma
  if seed is not None:
    np.random.seed(seed)
  init = init * (1 + sigma * np.random.randn(len(init)))
    
  return np.array(init)
  
class Worker(object):
  '''
  
  '''
  
  def __init__(self, koi, maxfun, debug):
    self.koi = koi
    self.maxfun = maxfun
    self.debug = debug
  
  def __call__(self, tag):
    
    # Load the data (if necessary)
    global data
    if data is None:
      data = GetData(self.koi, data_type = 'bkg')
    
    if self.debug:
      print("Began decorrelation for tag", tag)
  
    i = tag[0]
    q = tag[1]
    quarter = data[q]
  
    if quarter['time'] == []:
      if self.debug: 
        print("No data for tag", tag)
      return False
  
    # Decorrelate
    init = InitialGuess(self.koi, q, seed = i)
    
    if self.debug:
      print("Initial guess for tag", tag, ":", init)
    
    res = Decorrelate(self.koi, q, init, debug = self.debug, maxfun = self.maxfun)
  
    # Save
    if not os.path.exists(os.path.join(datadir, str(self.koi), 'pld')):
      os.makedirs(os.path.join(datadir, str(self.koi), 'pld'))
    np.savez(os.path.join(datadir, str(self.koi), 'pld', '%02d.%02d.npz' % (q, i)), **res)
  
    if self.debug:
      print("Decorrelation finished for tag", tag)
  
    return True  
  
def Run(koi = 254.01, quarters = list(range(18)), tag = 0, maxfun = 15000, debug = False, pool = None):
  '''
  
  '''
  
  # Multiprocess?
  if pool is None:
    M = map
  else:
    M = pool.map
  
  # Set up our list of runs  
  tags = [(tag, q) for q in quarters]
  W = Worker(koi, maxfun, debug)
  
  # Run!
  return list(M(W, tags))

def Plot(koi = 254.01, quarters = list(range(18))):
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
    
      # Look at the likelihoods of all runs for this quarter
      pldpath = os.path.join(datadir, str(koi), 'pld')
      files = [os.path.join(pldpath, f) for f in os.listdir(pldpath) if f.startswith('%02d.' % q) and f.endswith('.npz')]
      
      if len(files) == 0:
        continue
      
      lnl = np.zeros_like(files)
      for i, f in enumerate(files):
        lnl[i] = float(np.load(f)['lnlike'])
      
      # Grab the highest likelihood run
      res = np.load(files[np.argmax(lnl)])
      time = res['time']
      fsum = res['fsum']
      pmod = res['pmod']
      gpmu = res['gpmu']
      yerr = res['yerr']
      coeffs = res['coeffs']
      lnlike = res['lnlike']
      info = res['info'][()]
    
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
    
  fig.savefig(os.path.join(datadir, str(koi), 'pld', 'decorr.png'), bbox_inches = 'tight')