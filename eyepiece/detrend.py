#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------


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

__all__ = ['Detrend', 'PlotDetrended', 'Whiten', 'PlotTransits', 'GetWhitenedData']

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

def NegLnLike(x, id, q, kernel, debug):
  '''
  Returns the negative log-likelihood for the model with coefficients ``x``,
  as well as its gradient with respect to ``x``.
  
  '''

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg')
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
    
def QuarterDetrend(id, q, kernel, init, bounds, maxfun, debug):
  '''
  
  '''
  
  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg')
  qd = data[q]
  
  # Run the optimizer.
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = False,
                      args = (id, q, kernel, debug), bounds = bounds,
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
  
  def __init__(self, id, kernel, kinit, sigma, kbounds, maxfun, debug, datadir):
    self.id = id
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
      data = GetData(self.id, data_type = 'bkg', datadir = self.datadir)
    
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
    res = QuarterDetrend(self.id, q, self.kernel, init, bounds, self.maxfun, self.debug)
  
    # Save
    if not os.path.exists(os.path.join(self.datadir, str(self.id), 'detrend')):
      os.makedirs(os.path.join(self.datadir, str(self.id), 'detrend'))
    np.savez(os.path.join(self.datadir, str(self.id), 'detrend', '%02d.%02d.npz' % (q, i)), **res)
  
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
  W = Worker(inp.id, inp.kernel, inp.kinit, inp.pert_sigma, 
             inp.kbounds, inp.maxfun, inp.debug, inp.datadir)
  
  # Run and save
  for res in M(W, tags):
    if not inp.quiet:
      print("Detrending complete for tag " + str(res[0]))
  
  return

def LoadBestRun(inp, q):
  '''
  
  '''
  
  # Attempt to load it, if it's been set already
  try:
    res = np.load(os.path.join(pldpath, "%02d.npz" % q))
  except:
    pass
  
  # Find it among all the runs for that quarter
  try:
  
    # Look at the likelihoods of all runs for this quarter
    pldpath = os.path.join(inp.datadir, str(inp.id), 'detrend')
    files = [os.path.join(pldpath, f) for f in os.listdir(pldpath) 
             if f.startswith('%02d.' % q) and f.endswith('.npz')]

    # Is there data this quarter?
    if len(files) == 0:
      return None

    # Grab the highest likelihood run
    lnl = np.zeros_like(files, dtype = float)
    for i, f in enumerate(files):
      lnl[i] = float(np.load(f)['lnlike'])
    res = np.load(files[np.argmax(lnl)])

    # Save this as the best one
    np.savez(os.path.join(pldpath, "%02d.npz" % q), **res)
    
    # Return
    return res
    
  except IOError:
    return None

def PlotDetrended(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  
  if not inp.quiet:
    print("Plotting detrended background flux...")
  
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
    res = LoadBestRun(inp, q)
    if res is None: 
      continue
  
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
        ax[1].annotate("\n" * (i + 2) + "   %.2f" % c, (ltq, yp1), ha='left', va='top', fontsize = 8)
      
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
  
  fig.savefig(os.path.join(inp.datadir, str(inp.id), 'detrended.png'), bbox_inches = 'tight')
  
  return fig, ax

def GetWhitenedData(input_file = None, folded = True):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  
  if not inp.quiet:
    print("Whitening the flux...")
    
  # Number of kernel params
  nkpars = len(inp.kernel.pars)

  # Load the data
  bkg = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  prc = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  info = GetInfo(inp.id, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']

  if folded and len(tN) == 0:
    raise Exception("No transits for current target!")

  # Whiten
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  for q in inp.quarters:

    # Load coefficients for this quarter
    res = LoadBestRun(inp, q)
    if res is None:
      if not inp.quiet:
        print("WARNING: No decorrelation info found for quarter %d." % q)
      continue
    else:
      x = res['x']
    
    for b_time, b_fpix, b_perr, time, fpix in zip(bkg[q]['time'], bkg[q]['fpix'], bkg[q]['perr'], prc[q]['time'], prc[q]['fpix']):
    
      # Whiten the flux
      flux = Whiten(x, b_time, b_fpix, b_perr, time, fpix, kernel = inp.kernel, crowding = prc[q]['crowding'])
    
      # Fold the time
      if folded:
        time -= np.array([tN[np.argmin(np.abs(tN - ti))] for ti in time])
    
      # Plot
      t = np.append(t, time)
      f = np.append(f, flux)
  
  return t, f

def PlotTransits(input_file = None):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)

  if not inp.quiet:
    print("Plotting transits...")

  # Load the info
  info = GetInfo(inp.id, datadir = inp.datadir); info.update(inp.info)
  tdur = info['tdur']
  t, f = GetWhitenedData(input_file, folded = True)

  # Plot
  fig, ax = pl.subplots(1, 1, figsize = inp.transits_figsize)
  xlim = (-inp.padtrn * tdur / 2., inp.padtrn * tdur / 2.)
  fvis = f[np.where((t > xlim[0]) & (t < xlim[1]))]
  minf = np.min(fvis)
  maxf = np.max(fvis)
  padf = 0.1 * (maxf - minf)
  ylim = (minf - padf, maxf + padf)
  ax.plot(t, f, 'k.', alpha = min(1.0, max(0.05, 375. / len(fvis))))
  
  # Bin to median
  bins = np.linspace(xlim[0], xlim[1], inp.tbins)
  delta = bins[1] - bins[0]
  idx  = np.digitize(t, bins)
  med = [np.median(f[idx == k]) for k in range(inp.tbins)]
  ax.plot(bins - delta / 2., med, 'ro', alpha = 0.5)
  ax.plot(bins - delta / 2., med, 'r-', lw = 2, alpha = 0.5)
  
  ax.set_xlim(xlim)  
  ax.set_ylim(ylim)
  ax.set_title('Folded Whitened Transits', fontsize = 24)
  ax.set_xlabel('Time (days)', fontsize = 22)
  ax.set_ylabel('Flux', fontsize = 22)
  fig.savefig(os.path.join(inp.datadir, str(inp.id), 'folded.png'), bbox_inches = 'tight')
  
  return fig, ax