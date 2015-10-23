#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------


.. todo::
   - Use highest likelihood run to whiten transits

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .download import GetData, GetInfo
from .interruptible_pool import InterruptiblePool
from .lnlike import LnLike
from .utils import Input
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as pl
import george
import os
import itertools

__all__ = ['Detrend', 'PlotDetrended', 'Whiten', 'PlotTransits', 'GetWhitenedTransits']

data = None

def Whiten(x, b_time, b_fpix, b_perr, time, fpix, kernel = 1. * george.kernels.Matern32Kernel(1.), crowding = None):
  '''
  
  '''
  
  # Calculate fsum
  b_fsum = np.sum(b_fpix, axis = 1)
  fsum = np.sum(fpix, axis = 1)
  
  # PLD coefficients
  c = x[len(kernel.pars):]
  
  # Kernel params
  kernel.pars = x[:len(kernel.pars)]
  
  # Number of background data points
  b_K = len(b_time)

  # The pixel model
  b_pmod = np.sum(b_fpix * np.outer(1. / b_fsum, c), axis = 1)
  pmod = np.sum(fpix * np.outer(1. / fsum, c), axis = 1)

  # Errors on detrended background data
  X = 1. + b_pmod / b_fsum
  B = X.reshape(b_K, 1) * b_perr - c * b_perr / b_fsum.reshape(b_K, 1)
  b_yerr = np.sum(B ** 2, axis = 1) ** 0.5

  # Compute the likelihood
  gp = george.GP(kernel)
  gp.compute(b_time, b_yerr)
  mu, _ = gp.predict(b_fsum - b_pmod, time)
  
  # The full decorrelated flux with baseline = 1.
  dflux = 1. + (fsum - mu - pmod) / fsum
  
  # Correct for crowding?
  if crowding is not None:
    dflux = (dflux - 1.) / crowding + 1.
  
  return dflux

def NegLnLike(x, koi, q, kernel, debug):
  '''
  Returns the negative log-likelihood for the model with coefficients ``x``,
  as well as its gradient with respect to ``x``.
  
  '''

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  dq = data[q]
  
  # The log-likelihood and its gradient
  ll = 0
  grad_ll = np.zeros_like(x, dtype = float)
  
  # Loop over each chunk in this quarter individually
  for time, fsum, fpix, perr in zip(dq['time'], dq['fsum'], dq['fpix'], dq['perr']):
    res = LnLike(x, time, fpix, perr, fsum = fsum, kernel = kernel)
    ll += res[0]
    grad_ll += res[1]
  
  # Check the progress by printing to the screen
  if debug:
    print(q, ll)
  
  return (-ll, -grad_ll)
    
def QuarterDetrend(koi, q, kernel, init, bounds, maxfun, debug):
  '''
  
  '''
  
  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(koi, data_type = 'bkg')
  qd = data[q]
  
  # Run the optimizer.
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = False,
                      args = (koi, q, kernel, debug), bounds = bounds,
                      m = 10, factr = 1.e1, pgtol = 1e-05, maxfun = maxfun)
  
  # Grab some info
  x = res[0]
  lnlike = -res[1]
  info = res[2]       

  # Compute a GP prediction
  time = np.array([], dtype = float)
  fsum = np.array([], dtype = float)
  ypld = np.array([], dtype = float)
  gpmu = np.array([], dtype = float)
  gperr = np.array([], dtype = float)
  
  for t, fs, fp, pe in zip(qd['time'], qd['fsum'], qd['fpix'], qd['perr']):
    res = LnLike(x, t, fp, pe, fsum = fs, kernel = kernel, predict = True)
    time = np.append(time, t)
    fsum = np.append(fsum, fs)
    ypld = np.append(ypld, res[2])
    gpmu = np.append(gpmu, res[3])
    gperr = np.append(gperr, res[4])

  return {'time': time, 'fsum': fsum, 'ypld': ypld, 'gpmu': gpmu, 'gperr': gperr,
          'x': x, 'lnlike': lnlike, 'info': info, 'init': init}
  
class Worker(object):
  '''
  
  '''
  
  def __init__(self, koi, kernel, kinit, sigma, kbounds, maxfun, debug, datadir):
    self.koi = koi
    self.kernel = kernel
    self.kinit = kinit
    self.sigma = sigma
    self.kbounds = kbounds
    self.maxfun = maxfun
    self.debug = debug
    self.datadir = datadir
  
  def __call__(self, tag):
    
    # Load the data (if necessary)
    global data
    if data is None:
      data = GetData(self.koi, data_type = 'bkg', datadir = self.datadir)
    
    # Tags: i is the iteration number; q is the quarter number
    i = tag[0]
    q = tag[1]
    qd = data[q]
  
    # Is there data for this quarter?
    if qd['time'] == []:
      return (tag, False)
  
    # Set our initial guess
    npix = qd['fpix'][0].shape[1]
    init = np.append(self.kinit, [np.median(qd['fsum'][0])] * npix)
    bounds = np.concatenate([self.kbounds, [[-np.inf, np.inf]] * npix])
  
    # Perturb initial conditions by sigma, and ensure within bounds
    np.random.seed(tag)
    foo = bounds[:,0]
    while np.any(foo <= bounds[:,0]) or np.any(foo >= bounds[:,1]):
      foo = init * (1 + self.sigma * np.random.randn(len(init)))
    init = foo
    
    # Detrend
    res = QuarterDetrend(self.koi, q, self.kernel, init, bounds, self.maxfun, self.debug)
  
    # Save
    if not os.path.exists(os.path.join(self.datadir, str(self.koi), 'detrend')):
      os.makedirs(os.path.join(self.datadir, str(self.koi), 'detrend'))
    np.savez(os.path.join(self.datadir, str(self.koi), 'detrend', '%02d.%02d.npz' % (q, i)), **res)
  
    return (tag, True) 
  
def Detrend(input_file = None, pool = None):

  '''

  '''
  
  # Load inputs
  inp = Input(input_file)

  # Multiprocess?
  if pool is None:
    M = map
  else:
    M = pool.map

  # Set up our list of runs  
  tags = list(itertools.product(range(inp.niter), inp.quarters))
  W = Worker(inp.koi, inp.kernel, inp.kinit, inp.pert_sigma, 
             inp.kbounds, inp.maxfun, inp.debug, inp.datadir)
  
  # Run and save
  for res in M(W, tags):
    if not inp.quiet:
      print("Detrending complete for tag " + str(res[0]))
  
  return

def PlotDetrended(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  
  # Number of kernel params
  nkpars = len(inp.kernel.pars)
  
  # Plot the decorrelated data
  fig, ax = pl.subplots(3, 1, figsize = inp.detrend_figsize) 
  
  # Some miscellaneous info
  lt = [None] * (inp.quarters[-1] + 1)
  wf = [""] * (inp.quarters[-1] + 1)
  fc = np.zeros(inp.quarters[-1] + 1)
  ni = np.zeros(inp.quarters[-1] + 1)
  ll = np.zeros(inp.quarters[-1] + 1)
  cc = [None] * (inp.quarters[-1] + 1)
  for q in inp.quarters:
    
    # Load the decorrelated data
    try:
    
      # Look at the likelihoods of all runs for this quarter
      pldpath = os.path.join(inp.datadir, str(inp.koi), 'detrend')
      files = [os.path.join(pldpath, f) for f in os.listdir(pldpath) 
               if f.startswith('%02d.' % q) and f.endswith('.npz')]
      
      # Is there data this quarter?
      if len(files) == 0:
        continue
      
      # Grab the highest likelihood run
      lnl = np.zeros_like(files, dtype = float)
      for i, f in enumerate(files):
        lnl[i] = float(np.load(f)['lnlike'])
      res = np.load(files[np.argmax(lnl)])
      
      # Save this as the best one
      np.savez(os.path.join(pldpath, "%02d.npz" % q), **res)
      
      # Grab the detrending info
      time = res['time']
      fsum = res['fsum']
      ypld = res['ypld']
      gpmu = res['gpmu']
      gperr = res['gperr']
      x = res['x']
      lnlike = res['lnlike']
      info = res['info'][()]
      init = res['init']
    
    except IOError:
      continue
    
    # The SAP flux
    ax[0].plot(time, fsum, 'k.', alpha = 0.3)
    
    # The PLD-detrended SAP flux (blue) and the GP (red)
    ax[1].plot(time, ypld, 'b.', alpha = 0.3)
    ax[1].plot(time, gpmu, 'r-')
  
    # The fully detrended flux
    f = ypld - gpmu
    ax[2].plot(time, f, 'b.', alpha = 0.3)
    ax[2].fill_between(time, f - gperr, f + gperr, alpha = 0.1, lw = 0, color = 'r')
    
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
    cc[q] = x
      
  ltq = ax[0].get_xlim()[0]  
  yp0 = ax[0].get_ylim()[1]
  yp1 = ax[1].get_ylim()[1]
  yp2 = ax[2].get_ylim()[1]
  
  for q in inp.quarters:
    
    # This stores the last timestamp of the quarter
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
  
  fig.savefig(os.path.join(inp.datadir, str(inp.koi), 'detrended.png'), bbox_inches = 'tight')
  
  return fig, ax

def GetWhitenedTransits(input_file = None):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  
  # Number of kernel params
  nkpars = len(inp.kernel.pars)

  # Load the data
  bkg = GetData(inp.koi, data_type = 'bkg', datadir = inp.datadir)
  prc = GetData(inp.koi, data_type = 'prc', datadir = inp.datadir)
  tN, per, tdur, hash = GetInfo(inp.koi, datadir = inp.datadir)

  # Whiten
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  for q in inp.quarters:

    # Load coefficients for this quarter
    try:
      x = np.load(os.path.join(inp.datadir, str(inp.koi), 'detrend', '%02d.npz' % q))['x']
    except:
      if not inp.quiet:
        print("WARNING: No decorrelation info found for quarter %d." % q)
      continue
    
    for b_time, b_fpix, b_perr, time, fpix in zip(bkg[q]['time'], bkg[q]['fpix'], bkg[q]['perr'], prc[q]['time'], prc[q]['fpix']):
    
      # Whiten the flux
      flux = Whiten(x, b_time, b_fpix, b_perr, time, fpix, kernel = inp.kernel, crowding = prc[q]['crowding'])
    
      # Fold the time
      time -= [tN[np.argmin(np.abs(tN - t))] for t in time]
    
      # Plot
      t = np.append(t, time)
      f = np.append(f, flux)
  
  return t, f

def PlotTransits(input_file = None):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)

  # Load the info
  tN, per, tdur, hash = GetInfo(inp.koi, datadir = inp.datadir)
  t, f = GetWhitenedTransits(input_file)

  # Plot
  fig, ax = pl.subplots(1, 1, figsize = inp.transits_figsize)
  ax.plot(t, f, 'k.', alpha = 0.1)
  ax.set_xlim(-inp.padtrn * tdur / 2., inp.padtrn * tdur / 2.)  
  ax.set_title('Folded Whitened Transits', fontsize = 24)
  fig.savefig(os.path.join(inp.datadir, str(inp.koi), 'folded.png'), bbox_inches = 'tight')
  
  return fig, ax