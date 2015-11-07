#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------

.. todo::
   - Save ``input.py`` file for each target

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .download import GetData, GetInfo, GetPDCFlux
from .interruptible_pool import InterruptiblePool
from .linalg import LnLike, Whiten
from .utils import Input, GetOutliers, FunctionWrapper, GitHash
from .inspect import Inspect
import numpy as np; np.seterr(invalid = 'ignore')
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as pl
import george
import os
import itertools
from scipy.optimize import curve_fit

__all__ = ['Detrend', 'PlotDecorrelation', 'GetWhitenedData', 'Compare', 'PlotTransits', 'Plot']

data = None

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
    
def QuarterDetrend(tag, id, kernel, order, kinit, sigma, kbounds, maxfun, debug, datadir):
  '''
  
  '''

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg', datadir = datadir)
  
  # Tags: i is the iteration number; q is the quarter number
  i = tag[0]
  q = tag[1]
  qd = data[q]

  # Is there data for this quarter?
  if qd['time'] == []:
    return None

  # Set our initial guess
  npix = qd['fpix'][0].shape[1]
  if kernel is None and order is None:
    # Just PLD
    init = np.array([np.median(qd['fsum'][0])] * npix)
    bounds = np.array([[-np.inf, np.inf]] * npix)
  else:
    # PLD + GP or polynomial
    init = np.append(kinit, [np.median(qd['fsum'][0])] * npix)
    bounds = np.concatenate([kbounds, [[-np.inf, np.inf]] * npix])

  # Perturb initial conditions by sigma, and ensure within bounds
  np.random.seed(tag)
  foo = bounds[:,0]
  while np.any(foo <= bounds[:,0]) or np.any(foo >= bounds[:,1]):
    foo = init * (1 + sigma * np.random.randn(len(init)))
  init = foo
  
  # If we're doing PLD + polynomial, no need to run fmin_l_bfgs_b!
  if order is not None:
  
    # Concatenate the arrays for this quarter
    time = np.array([x for y in qd['time'] for x in y])
    fsum = np.array([x for y in qd['fsum'] for x in y])
    fpix = np.array([x for y in qd['fpix'] for x in y])
    perr = np.array([x for y in qd['perr'] for x in y])
    
    # Normalized time
    tnorm = ((time - time[0]) / (time[-1] - time[0]))
    
    iPLD = order + 1
    
    # Our pixel model
    def pm(y, *x):
      poly = np.sum([c * tnorm ** i for i, c in enumerate(x[:iPLD])], axis = 0)
      return poly + np.sum(fpix * np.outer(1. / fsum, x[iPLD:]), axis = 1)
    
    # Solve the (linear) problem
    x, _ = curve_fit(pm, None, fsum, p0 = init)
  
    # Here's our detrended data
    pixmod = np.sum(fpix * np.outer(1. / fsum, x[iPLD:]), axis = 1)
    ypld = fsum - pixmod
    gpmu = np.sum([c * tnorm ** i for i, c in enumerate(x[:iPLD])], axis = 0)
    
    # The new errors
    K = len(time)
    X = 1. + pixmod / fsum
    B = X.reshape(K, 1) * perr - x[iPLD:] * perr / fsum.reshape(K, 1)
    gperr = np.sum(B ** 2, axis = 1) ** 0.5
    
    return {'time': time, 'fsum': fsum, 'ypld': ypld, 'gpmu': gpmu, 'gperr': gperr,
            'x': x, 'lnlike': 0, 'init': init, 'tag': tag,
            'info': {'warnflag': 0, 'funcalls': 0, 'nit': 0, 'task': ''}, }
  
  else:
  
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
            'x': x, 'lnlike': lnlike, 'info': info, 'init': init, 'tag': tag}

def Detrend(input_file = None, pool = None):

  '''

  '''
  
  # Load inputs
  inp = Input(input_file)
  pldpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  allpath = os.path.join(pldpath, '_all')
  kernel = inp.kernel
  
  # Run inspection if needed
  success = Inspect(input_file)
  if not success:
    if not inp.quiet:
      print("Detrending aborted!")
    return False

  if not inp.quiet:
    print("Detrending...")

  # Set up our list of runs  
  tags = list(itertools.product(range(inp.niter), inp.quarters))
  FW = FunctionWrapper(QuarterDetrend, inp.id, inp.kernel, inp.order, inp.kinit, inp.pert_sigma, 
                       inp.kbounds, inp.maxfun, inp.debug, inp.datadir)
  
  # Parallelize?
  if pool is None:
    M = map
  else:
    M = pool.map
  
  # Run and save
  for res in M(FW, tags):
  
    # No data?
    if res is None:
      continue
  
    # Save
    if not os.path.exists(allpath):
      os.makedirs(allpath)
    np.savez(os.path.join(allpath, '%02d.%02d.npz' % res['tag'][::-1]), **res)
  
    # Print
    if not inp.quiet:
      print("Detrending complete for tag " + str(res['tag']))
  
  # We're going to update the trn data with the detrending info
  data = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  
  # Identify the highest likelihood run
  for q in inp.quarters:
  
    files = [os.path.join(allpath, f) for f in os.listdir(allpath) 
             if f.startswith('%02d.' % q) and f.endswith('.npz')]
    
    # Is there data this quarter?
    if len(files) == 0:
      return None

    # Grab the highest likelihood run
    lnl = np.zeros_like(files, dtype = float)
    for i, f in enumerate(files):
      lnl[i] = float(np.load(f)['lnlike'])
    res = np.load(files[np.argmax(lnl)])

    # Save this as the best run
    np.savez(os.path.join(pldpath, "%02d.npz" % q), **res)
  
    # Now detrend the transit data and save to disk
    gp_arr = []
    ypld_arr = []
    yerr_arr = []
    x = res['x']
    kernel.pars = x[:iPLD]
    c = x[iPLD:]
    for time, fpix, perr in zip(data[q]['time'], data[q]['fpix'], data[q]['perr']):
      fsum = np.sum(fpix, axis = 1)
      K = len(time)
      pixmod = np.sum(fpix * np.outer(1. / fsum, c), axis = 1)
      X = 1. + pixmod / fsum
      B = X.reshape(K, 1) * perr - c * perr / fsum.reshape(K, 1)
      yerr = np.sum(B ** 2, axis = 1) ** 0.5
      ypld = fsum - pixmod
      gp = george.GP(kernel)
      gp.compute(time, yerr)
    
      gp_arr.append(gp)
      ypld_arr.append(ypld)
      yerr_arr.append(yerr)
    
    # Update our transit file with the detrended data
    data[q].update({'dvec': x})
    data[q].update({'gp': gp_arr})
    data[q].update({'ypld': ypld_arr})
    data[q].update({'yerr': yerr_arr})
    
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'trn.npz'), data = data, hash = GitHash())  
    
  return True

def LoadQuarter(inp, q):
  '''
  
  '''
  
  pldpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  
  try:
    res = np.load(os.path.join(pldpath, "%02d.npz" % q))
    return res
  except IOError:
    return None

def PlotDecorrelation(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  
  if not inp.quiet:
    print("Plotting detrended background flux...")
  
  # Load some info
  info = GetInfo(inp.id, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']
  tdur = info['tdur']
  
  # Index of first PLD coefficient in ``x``
  if inp.kernel is not None:
    iPLD = len(inp.kernel.pars)
  else:
    if inp.order is None:
      iPLD = 0
    else:
      iPLD = inp.order + 1
  
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
    res = LoadQuarter(inp, q)
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
    
    # The PLD-detrended SAP flux (blue) and the GP/polynomial (red)
    if inp.kernel is not None:
      ax[1].plot(time, ypld, 'b.', alpha = 0.3)
      ax[1].plot(time, gpmu, 'r-')
    else:
      # Subtract off the median for better plotting
      ax[1].plot(time, ypld - np.nanmedian(ypld), 'b.', alpha = 0.3)
      ax[1].plot(time, gpmu - np.nanmedian(gpmu), 'r-')
  
    # The fully detrended flux
    f = ypld - gpmu
    ax[2].plot(time, f, 'b.', alpha = 0.3)
    ax[2].fill_between(time, f - gperr, f + gperr, alpha = 0.1, lw = 0, color = 'r')
    
    # Appearance
    [axis.set_xticklabels([]) for axis in ax[:-1]]
    [axis.margins(0, 0.01) for axis in ax]
    
    # Extra info
    lt[q] = time[-1]
    fc[q] = info['funcalls']
    ni[q] = info['nit']
    if info['warnflag']:
      wf[q] = str(info['task'])
      if 'ABNORMAL_TERMINATION_IN_LNSRCH' in wf[q]:
        wf[q] = 'AT_LNSRCH'
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
      for i, c in enumerate(cc[q][iPLD:]):
        ax[0].annotate("\n" * (i + 2) + "   %.1f" % c, (ltq, yp0), ha='left', va='top', fontsize = 8, color = 'r')
    
      # Best GP/polynomial param values
      if inp.kernel is not None:
        ax[1].annotate("\n   GP PARAMS", (ltq, yp1), ha='left', va='top', fontsize = 8)
      else:
        ax[1].annotate("\n   POLY COEFFS", (ltq, yp1), ha='left', va='top', fontsize = 8)
      for i, c in enumerate(cc[q][:iPLD]):
        ax[1].annotate("\n" * (i + 2) + "   %.2f" % c, (ltq, yp1), ha='left', va='top', fontsize = 8)
      
      # Optimization info
      if wf[q] == "":
        ax[2].annotate("\n   SUCCESS", (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'b')
      else:
        ax[2].annotate("\n   %s" % wf[q], (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'r')
      if inp.kernel is not None:
        ax[2].annotate("\n\n   CALLS: %d" % fc[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
        ax[2].annotate("\n\n\n   NITR: %d" % ni[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
        ax[2].annotate("\n\n\n\n   LNLK: %.2f" % ll[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
    
    for axis in ax:
      axis.axvline(lt[q], color='k', ls = '--')
    
    ltq = lt[q]
  
  # Labels and titles
  ax[0].set_title('Raw Background Flux', fontsize = 28, fontweight = 'bold', y = 1.1) 
  ax[1].set_title('PLD-Decorrelated Flux', fontsize = 28, fontweight = 'bold', y = 1.025)  
  if inp.kernel is not None:
    ax[2].set_title('PLD+GP-Decorrelated Flux', fontsize = 28, fontweight = 'bold', y = 1.025) 
  else:
    ax[2].set_title('PLD+POLY-Decorrelated Flux', fontsize = 28, fontweight = 'bold', y = 1.025) 
  ax[-1].set_xlabel('Time (Days)', fontsize = 24)
  [axis.set_ylabel('Counts', fontsize = 24) for axis in ax]
  
  # Appearance
  for s in ['top', 'bottom', 'left', 'right']:
    [axis.spines[s].set_linewidth(2) for axis in ax]
  [tick.label.set_fontsize(20) for tick in ax[-1].xaxis.get_major_ticks()]
  [tick.label.set_fontsize(18) for axis in ax for tick in axis.yaxis.get_major_ticks()] 
  ax[0].set_axis_bgcolor((1.0, 0.95, 0.95))
  ax[1].set_axis_bgcolor((0.95, 0.95, 0.95))
  ax[2].set_axis_bgcolor((0.95, 0.95, 1.0))
       
  fig.savefig(os.path.join(inp.datadir, str(inp.id), 'detrended.png'), bbox_inches = 'tight')
  
  return fig, ax

def GetWhitenedData(input_file = None, folded = True, return_mean = False):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  
  if not inp.quiet:
    print("Whitening the flux...")
    
  # Load the data
  bkg = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  prc = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  info = GetInfo(inp.id, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']
  per = info['per']

  if folded and len(tN) == 0:
    raise Exception("No transits for current target!")

  # Whiten
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  e = np.array([], dtype = float)
  for q in inp.quarters:

    # Load coefficients for this quarter
    res = LoadQuarter(inp, q)
    if res is None:
      if not inp.quiet:
        print("WARNING: No decorrelation info found for quarter %d." % q)
      continue
    else:
      x = res['x']
    
    for b_time, b_fpix, b_perr, time, fpix, perr in zip(bkg[q]['time'], bkg[q]['fpix'], bkg[q]['perr'], prc[q]['time'], prc[q]['fpix'], prc[q]['perr']):
    
      # Whiten the flux
      flux, ferr = Whiten(x, b_time, b_fpix, b_perr, time, fpix, perr, kernel = inp.kernel, order = inp.order, crowding = prc[q]['crowding'], return_mean = return_mean)
    
      # Fold the time
      if folded:
        time -= np.array([tN[np.argmin(np.abs(tN - ti))] for ti in time])
        
        # Remove tail preceding the very first transit
        bad = np.where(time < -per / 2.)
        time = np.delete(time, bad)
        flux = np.delete(flux, bad)  
        ferr = np.delete(ferr, bad)    
    
      t = np.append(t, time)
      f = np.append(f, flux)
      e = np.append(e, ferr)
  
  # Sort
  if folded:
    idx = np.argsort(t)
    t = t[idx]
    f = f[idx]
    e = e[idx]
  
  return t, f, e

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
  t, f, e = GetWhitenedData(input_file, folded = True)

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
  ax.plot(bins - delta / 2., med, 'ro', alpha = 0.75)
  
  ax.set_xlim(xlim)  
  ax.set_ylim(ylim)
  ax.set_title('Folded Whitened Transits', fontsize = 24)
  ax.set_xlabel('Time (days)', fontsize = 22)
  ax.set_ylabel('Flux', fontsize = 22)
  fig.savefig(os.path.join(inp.datadir, str(inp.id), 'folded.png'), bbox_inches = 'tight')
  
  return fig, ax
  
def Compare(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  
  if not inp.quiet:
    print("Detrending...")
  
  # Load some info
  info = GetInfo(inp.id, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']
  tdur = info['tdur']
    
  # Plot the decorrelated data
  fig, ax = pl.subplots(4, 1, figsize = (inp.detrend_figsize[0], inp.detrend_figsize[1] * 4. / 3.))     
  lt = [None] * (inp.quarters[-1] + 1)
  
  
  # --- PDC ---
  t, fpdc = GetPDCFlux(inp.id, inp.long_cadence)
  
  # Remove transits
  trnidx = np.array([], dtype = int)
  for tNi in tN:
    i = np.where(np.abs(tNi - t) <= tdur * inp.padbkg / 2.)[0]
    if len(i):
      trnidx = np.append(trnidx, i)
  t = np.delete(t, trnidx)
  fpdc = np.delete(fpdc, trnidx)

  # Plot
  ax[0].plot(t, fpdc, 'b.', alpha = 0.3)
  
  
  # --- PLD ---

  FW = FunctionWrapper(QuarterDetrend, inp.id, None, 5, [1.] * 6, inp.pert_sigma, 
                       [[-np.inf, np.inf]] * 6, inp.maxfun, inp.debug, inp.datadir)
  fpld = []
  for q in inp.quarters:
    res = FW((0,q))
    if res is None: 
      continue
    time = res['time']
    fsum = res['fsum']
    ypld = res['ypld']
    gpmu = res['gpmu']
    gperr = res['gperr']
    x = res['x']
    lnlike = res['lnlike']
    init = res['init']
    f = ypld - gpmu
    fpld.extend(f)
    ax[1].plot(time, f, 'b.', alpha = 0.3)
    ax[1].fill_between(time, f - gperr, f + gperr, alpha = 0.1, lw = 0, color = 'r')
  
    # Keep track of last time for each quarter for later
    lt[q] = time[-1]
    
  fpld = np.array(fpld)
  
  # --- GP ONLY ---
  data = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  fgp = []
  fgppld = []
  for q in inp.quarters:
    
    # Load the decorrelation results
    res = LoadQuarter(inp, q)
    if res is None: 
      continue
    
    # Load the data
    time = np.array([x for foo in data[q]['time'] for x in foo])
    fsum = np.array([x for foo in data[q]['fsum'] for x in foo])
    yerr = np.array([x for foo in data[q]['perr'] for x in foo])
    yerr = np.sqrt(np.sum(yerr ** 2, axis = 1))
    
    # GP
    inp.kernel.pars = res['x'][:len(inp.kernel.pars)]
    gp = george.GP(inp.kernel)
    gp.compute(time, yerr)
    mu, _ = gp.predict(fsum - np.median(fsum), time)
    mu += np.median(fsum)
    fgp.extend(fsum - mu)
    
    # Plot
    ax[2].plot(time, fsum - mu, 'b.', alpha = 0.3)

    # Now plot the GP + PLD solution
    f = np.array(res['ypld']) - np.array(res['gpmu'])
    fgppld.extend(f)
    ax[3].plot(time, f, 'b.', alpha = 0.3)

  fgp = np.array(fgp)
  fgppld = np.array(fgppld)
  
  # Scale y limits
  out, M, MAD = GetOutliers(np.concatenate([fpdc, fgp, fgppld, fpld]), sig_tol = 5.)
  ax[0].set_ylim(M - 5 * MAD, M + 5 * MAD)
  ax[1].set_ylim(M - 5 * MAD, M + 5 * MAD)
  ax[2].set_ylim(M - 5 * MAD, M + 5 * MAD)
  ax[3].set_ylim(M - 5 * MAD, M + 5 * MAD)
  
  # --- Appearance ---
  [axis.set_xticklabels([]) for axis in ax[:-1]]
  [axis.margins(0, 0.01) for axis in ax]
  ltq = ax[0].get_xlim()[0]
  yp0 = ax[0].get_ylim()[1]
  for q in inp.quarters:
      
    # This stores the last timestamp of the quarter
    if lt[q] is None:
      continue
    
    # If the data spans more than 20 days, plot some info (otherwise, it won't fit!)
    if lt[q] - ltq > 20:
    
      # Quarter number
      ax[0].annotate(q, ((ltq + lt[q]) / 2., yp0), ha='center', va='bottom', fontsize = 24)
    
    # Quarter markers
    for axis in ax:
      axis.axvline(lt[q], color='k', ls = '--')
    
    ltq = lt[q]
  
  # Labels and titles
  ax[0].set_title('Kepler PDC Flux', fontsize = 28, fontweight = 'bold', y = 1.1)
  ax[1].set_title('PLD + Polynomial', fontsize = 28, fontweight = 'bold', y = 1.025) 
  ax[2].set_title('GP Only', fontsize = 28, fontweight = 'bold', y = 1.025)  
  ax[3].set_title('Templar', fontsize = 28, fontweight = 'bold', y = 1.025)
  ax[3].set_xlabel('Time (Days)', fontsize = 24)
  [axis.set_ylabel('Counts', fontsize = 24) for axis in ax]
  
  # Appearance
  for s in ['top', 'bottom', 'left', 'right']:
    [axis.spines[s].set_linewidth(2) for axis in ax]
  [tick.label.set_fontsize(20) for tick in ax[-1].xaxis.get_major_ticks()]
  [tick.label.set_fontsize(18) for axis in ax for tick in axis.yaxis.get_major_ticks()] 
  ax[0].set_axis_bgcolor((1., 0.95, 0.95))
  ax[1].set_axis_bgcolor((1., 0.95, 0.95))
  ax[2].set_axis_bgcolor((1., 0.95, 0.95))
  ax[3].set_axis_bgcolor((0.9, 0.9, 1.0))


  # Save
  fig.savefig(os.path.join(inp.datadir, str(inp.id), 'comparison.png'), bbox_inches = 'tight')
  
  return fig, ax

def Plot(input_file = None):
  '''
  
  '''
  
  PlotDecorrelation(input_file)
  PlotTransits(input_file)