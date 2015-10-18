#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------


'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .download import GetData
from .config import datadir
from .interruptible_pool import InterruptiblePool
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as pl
import george
import os

__all__ = ['Detrend', 'PlotDetrended']
data = None

def NegLnLike(coeffs, koi, q, kernel, pld):
  '''
  Returns the negative log-likelihood for the model with coefficients ``coeffs``.
  If PLD is False, also returns the gradient of the log-likelihood with respect
  to the kernel parameters.
  
  '''

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  quarter = data[q]

  # Our quasi-periodic kernel
  nkpars = len(kernel.pars)
  kernel.pars = coeffs[:nkpars]
  gp = george.GP(kernel)
  
  # The log-likelihood
  ll = 0
  
  # Its gradient
  grad_ll = np.zeros_like(coeffs, dtype = float)
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):
  
    if pld:
    
      # The pixel model
      pmod = np.sum(fpix * np.outer(1. / fsum, coeffs[nkpars:]), axis = 1)
    
      # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
      T = np.ones_like(fsum)
      ferr = np.zeros_like(fsum)
      for k, _ in enumerate(ferr):
        ferr[k] = np.sqrt(np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (coeffs[nkpars:] / fsum[k])) ** 2 * perr[k] ** 2))
    
    else:
      
      # The error is just the sum in quadrature of the pixel errors
      ferr = np.sqrt(np.sum(perr ** 2, axis = 1))

      # Our "pixel model" is just the median flux
      pmod = np.ones_like(fsum) * np.median(fsum)
    
    # Evaluate the model
    try:      

      # Compute the likelihood
      gp.compute(time, ferr)
      ll += gp.lnlikelihood(fsum - pmod)
    
      # If we're only doing GP, compute the gradient
      # Note that george return d (ln Like) / d (ln X), so we need to divide by X
      if not pld:
        grad_ll += gp.grad_lnlikelihood(fsum - pmod) / kernel.pars
      
    except Exception as e:

      # Return a low likelihood
      ll = -1.e10
      grad_ll = np.zeros_like(coeffs, dtype = float)
      break

  if not pld:
    return (-ll, -grad_ll)
  else:
    return -ll
    
def QuarterDetrend(koi, q, kernel, init, bounds, maxfun, pld):
  '''
  
  '''
  
  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  quarter = data[q]
  
  # Run the optimizer. We compute the gradient numerically if pld is True.
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = pld,
                      args = (koi, q, kernel, pld), bounds = bounds,
                      m = 10, factr = 1.e1, epsilon = 1e-8,
                      pgtol = 1e-05, maxfun = maxfun)
  
  coeffs = res[0]
  lnlike = -res[1]
  info = res[2]       

  all_time = []
  all_fsum = []
  all_pmod = []
  all_gpmu = []
  all_ferr = []
  all_yerr = []
  
  if kernel is not None:
    # Our quasi-periodic kernel
    nkpars = len(kernel.pars)
    kernel.pars = coeffs[:nkpars]
    gp = george.GP(kernel)
  
  else:
    # We're just going to fit a quadratic
    nkpars = 3
    a, b, c = coeffs[:nkpars]
  
  for time, fsum, fpix, perr in zip(quarter['time'], quarter['fsum'], quarter['fpix'], quarter['perr']):

    if pld:
    
      # The pixel model
      pmod = np.sum(fpix * np.outer(1. / fsum, coeffs[nkpars:]), axis = 1)
    
      # Propagate errors correctly. ``T`` is the transit model, which is all 1's here since we're masking the transits
      T = np.ones_like(fsum)
      ferr = np.zeros_like(fsum)
      for k, _ in enumerate(ferr):
        ferr[k] = np.sqrt(np.sum(((1. / T[k]) + (pmod[k] / fsum[k]) - (coeffs[nkpars:] / fsum[k])) ** 2 * perr[k] ** 2))

    else:
      
      # The error is just the sum in quadrature of the pixel errors
      ferr = np.sqrt(np.sum(perr ** 2, axis = 1))

      # Our "pixel model" is just the median flux
      pmod = np.ones_like(fsum) * np.median(fsum)

    # Model prediction (for plotting)
    gp.compute(time, ferr)
    mu, cov = gp.predict(fsum - pmod, time)
    yerr = np.sqrt(np.diag(cov))
    
    all_time.extend(time)
    all_fsum.extend(fsum)
    all_pmod.extend(pmod)
    all_gpmu.extend(mu)
    all_ferr.extend(ferr)
    all_yerr.extend(yerr)
    
  time = np.array(all_time)
  fsum = np.array(all_fsum)
  pmod = np.array(all_pmod)
  gpmu = np.array(all_gpmu)
  ferr = np.array(all_ferr)  
  yerr = np.array(all_yerr) 
  
  return {'time': time, 'fsum': fsum, 'pmod': pmod, 'gpmu': gpmu, 'ferr': ferr, 'yerr': yerr, 'coeffs': coeffs, 'lnlike': lnlike, 'info': info, 'init': init}
  
class Worker(object):
  '''
  
  '''
  
  def __init__(self, koi, kernel, kinit, sigma, kbounds, maxfun, pld):
    self.koi = koi
    self.kernel = kernel
    self.kinit = kinit
    self.sigma = sigma
    self.kbounds = kbounds
    self.maxfun = maxfun
    self.pld = pld
  
  def __call__(self, tag):
    
    # Load the data (if necessary)
    global data
    if data is None:
      data = GetData(self.koi, data_type = 'bkg')
  
    i = tag[0]
    q = tag[1]
    quarter = data[q]
  
    if quarter['time'] == []:
    
      # No data this quarter
      return False
  
    if self.pld:
      npix = quarter['fpix'][0].shape[1]
      init = np.append(self.kinit, [np.median(quarter['fsum'][0])] * npix)
      bounds = np.concatenate([self.kbounds, [[-np.inf, np.inf]] * npix])
    else:
      init = np.array(self.kinit)
      bounds = np.array(self.kbounds)
  
    # Perturb initial conditions by sigma
    np.random.seed(tag)
    foo = bounds[:,0]
    while np.any(foo <= bounds[:,0]) or np.any(foo >= bounds[:,1]):
      foo = init * (1 + self.sigma * np.random.randn(len(init)))
    init = foo
    
    # Detrend
    res = QuarterDetrend(self.koi, q, self.kernel, init, bounds, self.maxfun, self.pld)
  
    # Save
    if not os.path.exists(os.path.join(datadir, str(self.koi), 'pld')):
      os.makedirs(os.path.join(datadir, str(self.koi), 'pld'))
    np.savez(os.path.join(datadir, str(self.koi), 'pld', '%02d.%02d.npz' % (q, i)), **res)
  
    return True  
  
def Detrend(koi = 17.01, kernel = 1. * george.kernels.Matern32Kernel(1.), 
            quarters = list(range(18)), tag = 0, maxfun = 15000, pld = True, 
            sigma = 0.25, kinit = [100., 100.], kbounds = [[1.e-8, 1.e8], [1.e-4, 1.e8]], 
            pool = InterruptiblePool(), quiet = False):
  '''

  '''

  # Multiprocess?
  if pool is None:
    M = map
  else:
    M = pool.map

  # Set up our list of runs  
  tags = [(tag, q) for q in quarters]
  W = Worker(koi, kernel, kinit, sigma, kbounds, maxfun, pld)
  
  # Run and save
  res = list(M(W, tags))
  
  if not quiet: print("Detrending complete for tag %d." % tag)
  
  return res

def PlotDetrended(koi = 17.01, quarters = list(range(18)), kernel = 1. * george.kernels.Matern32Kernel(1.)):
  '''
  
  '''
  
  #
  if kernel is not None:
    nkpars = len(kernel.pars)
  else:
    nkpars = 3
  
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
      
      lnl = np.zeros_like(files, dtype = float)
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
      init = res['init']
    
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
  yp1 = ax[1].get_ylim()[1]
  yp2 = ax[2].get_ylim()[1]
  for q in quarters:  
    if lt[q] is None:
      continue
    
    # If the data spans more than 20 days, plot some info (otherwise, it won't fit!)
    if lt[q] - ltq > 20:
    
      # Quarter number
      ax[0].annotate(q, ((ltq + lt[q]) / 2., yp0), ha='center', va='bottom', fontsize = 24)
    
      # Best coeff values
      ax[0].annotate("\n   PLD COEFFS", (ltq, yp0), ha='left', va='top', fontsize = 8, color = 'r')
      for i, c in enumerate(cc[q][nkpars:]):
        ax[0].annotate("\n" * (i + 2) + "   %.1f" % c, (ltq, yp0), ha='left', va='top', fontsize = 8, color = 'r')
    
      # Best GP param values
      ax[1].annotate("\n   GP PARAMS", (ltq, yp1), ha='left', va='top', fontsize = 8)
      for i, c in enumerate(cc[q][:nkpars]):
        ax[1].annotate("\n" * (i + 2) + "   %.1f" % c, (ltq, yp1), ha='left', va='top', fontsize = 8)
      
      # Optimization info
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
  
  ax[0].set_title('Raw Background Flux', fontsize = 24, y = 1.1) 
  ax[1].set_title('PLD-Decorrelated Flux', fontsize = 24)  
  ax[2].set_title('PLD+GP-Decorrelated Flux', fontsize = 24)   
  fig.savefig(os.path.join(datadir, str(koi), 'pld', 'decorr.png'), bbox_inches = 'tight')