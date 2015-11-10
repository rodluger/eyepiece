#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
compare.py
----------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Input, GetData, GetOutliers, FreedmanDiaconis, FunctionWrapper
from scipy.optimize import curve_fit, fmin_l_bfgs_b
from scipy.stats import norm, normaltest
import matplotlib.mlab as mlab
import numpy as np
import os
import george
import matplotlib.pyplot as pl

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

data = None

def NegLnLikeGP(x, t, y, e, kernel):
  '''
  
  '''
  
  # The log-likelihood and its gradient
  ll = 0
  grad_ll = np.zeros_like(x, dtype = float)
  
  # The GP object
  kernel.pars = x
  gp = george.GP(kernel)
  
  # Loop over each chunk in this quarter individually
  for ti, yi, ei in zip(t, y, e):
    
    # Compute the likelihood
    try:
      gp.compute(ti, ei)
      ll += gp.lnlikelihood(yi)
    except:
      return (-np.inf, np.zeros_like(x))
    
    # Compute the gradient
    # Note that george returns dLnLike/dLnX, so we divide by X to get dLnLike/dX.
    grad_ll += gp.grad_lnlikelihood(yi) / kernel.pars
  
  return (-ll, -grad_ll)

def PDC(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)

  # Load the data
  data = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)

  # Return
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  for q in inp.quarters:
    for time, flux in zip(data[q]['time'], data[q]['pdcf']):
      t = np.append(t, time)
      f = np.append(f, flux)
  return t, f

def PLDPoly(input_file = None, order = 5):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)

  # Load the data
  data = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  
  for q in inp.quarters:
  
    qd = data[q]
    
    if len(qd['time']) == 0:
      continue
  
    # Concatenate the arrays for this quarter
    time = np.array([x for y in qd['time'] for x in y])
    fpix = np.array([x for y in qd['fpix'] for x in y])
    fsum = np.sum(fpix, axis = 1)
    npix = fpix.shape[1]
    init = np.array([1.] * (order + 1) + [np.median(fsum)] * npix)
    perr = np.array([x for y in qd['perr'] for x in y])
    
    # Normalized time
    tnorm = ((time - time[0]) / (time[-1] - time[0]))
    
    # PLD coeffs start index
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
    poly = np.sum([c * tnorm ** i for i, c in enumerate(x[:iPLD])], axis = 0)
    
    t = np.append(t, time)
    f = np.append(f, ypld - poly)
  
  return t, f

def PLD_then_GP(q, id, kernel, kinit, kbounds, maxfun, datadir):
  '''
  
  '''
  
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg', datadir = datadir)
  qd = data[q]

  if qd['time'] == []:
    return None

  # t/y for all chunks
  t_all = np.array([], dtype = float)
  y_all = np.array([], dtype = float)  
  
  # Solve the (linear) PLD problem for this quarter
  fpix = np.array([x for y in qd['fpix'] for x in y])
  fsum = np.sum(fpix, axis = 1)
  init = np.array([np.median(fsum)] * fpix.shape[1])
  def pm(y, *x):
    return np.sum(fpix * np.outer(1. / fsum, x), axis = 1)
  x, _ = curve_fit(pm, None, fsum, p0 = init)
  
  # Here's our detrended data
  y = []
  t = []
  e = []
  for time, fpix, perr in zip(qd['time'], qd['fpix'], qd['perr']):
    
    # The pixel model
    fsum = np.sum(fpix, axis = 1)
    pixmod = np.sum(fpix * np.outer(1. / fsum, x), axis = 1)
    
    # The errors
    X = np.ones_like(time) + pixmod / fsum
    B = X.reshape(len(time), 1) * perr - x * perr / fsum.reshape(len(time), 1)
    yerr = np.sum(B ** 2, axis = 1) ** 0.5

    # Append to our arrays
    y.append(fsum - pixmod)
    t.append(time)
    e.append(yerr)
  
  # Now we solve for the best GP params
  res = fmin_l_bfgs_b(NegLnLikeGP, kinit, approx_grad = False,
                      args = (t, y, e, kernel), bounds = kbounds,
                      m = 10, factr = 1.e1, pgtol = 1e-05, maxfun = maxfun)
  
  # Finally, detrend the data
  kernel.pars = res[0]
  gp = george.GP(kernel)
  for ti, yi, ei in zip(t, y, e):
    gp.compute(ti, ei)
    mu, _ = gp.predict(yi, ti)
    y_all = np.append(y_all, yi - mu)
    t_all = np.append(t_all, ti)
  
  return t_all, y_all

def GPOnly(q, id, kernel, kinit, kbounds, maxfun, datadir):
  '''
  
  '''
  
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg', datadir = datadir)
  qd = data[q]

  if qd['time'] == []:
    return None
  
  # t/y for all chunks
  t_all = np.array([], dtype = float)
  y_all = np.array([], dtype = float)

  # Solve for the best GP params
  t = qd['time']
  y = np.array([np.sum(fpix, axis = 1) - np.median(np.sum(fpix, axis = 1)) for fpix in qd['fpix']])
  e = [np.sqrt(np.sum(perr ** 2, axis = 1)) for perr in qd['perr']]

  res = fmin_l_bfgs_b(NegLnLikeGP, kinit, approx_grad = False,
                      args = (t, y, e, kernel), bounds = kbounds,
                      m = 10, factr = 1.e1, pgtol = 1e-05, maxfun = maxfun)
  
  # Finally, detrend the data
  kernel.pars = res[0]
  gp = george.GP(kernel)
  for ti, yi, ei in zip(t, y, e):
    gp.compute(ti, ei)
    mu, _ = gp.predict(yi, ti)
    y_all = np.append(y_all, yi - mu)
    t_all = np.append(t_all, ti)
  
  return t_all, y_all

def Templar(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  
  for q in inp.quarters:
    # Load the decorrelated data
    files = [os.path.join(detpath, file) for file in os.listdir(detpath) 
             if file.startswith('%02d.' % q) and file.endswith('.npz')]
  
    # Is there data this quarter?
    if len(files) == 0:
      continue

    # Grab the highest likelihood run
    lnl = np.zeros_like(files, dtype = float)
    for i, file in enumerate(files):
      lnl[i] = float(np.load(file)['lnlike'])
    res = np.load(files[np.argmax(lnl)])

    # Grab the detrending info
    time = res['time']
    ypld = res['ypld']
    gpmu = res['gpmu']
    
    t = np.append(t, time)
    f = np.append(f, ypld - gpmu)
  
  return t, f

def Compare(input_file = None, pool = None):
  '''
  
  '''
  
  # Reset the global variable
  global data
  data = None
  
  # Load
  inp = Input(input_file)
  datapath = os.path.join(inp.datadir, str(inp.id), '_data')
  
  if not inp.clobber:
    try:
      res = np.load(os.path.join(datapath, 'cmp.npz'))
      return
    except FileNotFoundError:
      pass
  
  if not inp.quiet:
    print("Running comparison...")
  
  # These are quick; no need to parallelize
  t, fPDC = PDC(input_file)
  _, fPLD = PLDPoly(input_file)
  _, fTem = Templar(input_file)
  
  # This one is slow
  fPLDGP = np.array([], dtype = float)
  FW = FunctionWrapper(PLD_then_GP, inp.id, inp.kernel, inp.kinit, 
                       inp.kbounds, inp.maxfun, inp.datadir)
  if pool is None: 
    M = map
  else: 
    M = pool.map
  for res in M(FW, inp.quarters):
    if res is None:
      continue
    fPLDGP = np.append(fPLDGP, res[1])
  
  # This one too
  fGP = np.array([], dtype = float)
  FW = FunctionWrapper(GPOnly, inp.id, inp.kernel, inp.kinit, 
                       inp.kbounds, inp.maxfun, inp.datadir)
  if pool is None: 
    M = map
  else: 
    M = pool.map
  for res in M(FW, inp.quarters):
    if res is None:
      continue
    fGP = np.append(fGP, res[1])

  # Save
  np.savez(os.path.join(datapath, 'cmp.npz'), t = t, fPDC = fPDC, fPLD = fPLD, fPLDGP = fPLDGP, fGP = fGP, fTem = fTem)

def PlotComparison(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  data = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  plotpath = os.path.join(inp.datadir, str(inp.id), '_plots')
  datapath = os.path.join(inp.datadir, str(inp.id), '_data')
   
  # Have we done this already?
  if not inp.clobber:
    if os.path.exists(os.path.join(plotpath, 'comparison.png')):
      return None, None
      
  # Load detrended data
  try:
    det = np.load(os.path.join(datapath, 'cmp.npz'))
  except FileNotFoundError:
    raise Exception("Please run ``Compare()`` first.")
  
  t = det['t']
  fPDC = det['fPDC']
  fPLD = det['fPLD']
  fPLDGP = det['fPLDGP']
  fGP = det['fGP']
  fTem = det['fTem']
  
  if not inp.quiet:
    print("Plotting detrending comparison...")
  
  fig = pl.figure()
  fig.set_size_inches(48,28)
  fig.subplots_adjust(wspace = 0.1, hspace = 0.3)
  
  ax0 = pl.subplot2grid((5,9), (0,0), colspan=8)
  ax0h = pl.subplot2grid((5,9), (0,8))
  
  ax1 = pl.subplot2grid((5,9), (1,0), colspan=8)
  ax1h = pl.subplot2grid((5,9), (1,8))
  
  ax2 = pl.subplot2grid((5,9), (2,0), colspan=8)
  ax2h = pl.subplot2grid((5,9), (2,8))
  
  ax3 = pl.subplot2grid((5,9), (3,0), colspan=8)
  ax3h = pl.subplot2grid((5,9), (3,8))
  
  ax4 = pl.subplot2grid((5,9), (4,0), colspan=8)
  ax4h = pl.subplot2grid((5,9), (4,8))
  
  ax = [ax0, ax1, ax2, ax3, ax4]
  axh = [ax0h, ax1h, ax2h, ax3h, ax4h]
  
  # Plot data   
  ax[0].plot(t, fPDC, 'b.', alpha = 0.3)
  ax[1].plot(t, fPLD, 'b.', alpha = 0.3)
  ax[2].plot(t, fPLDGP, 'b.', alpha = 0.3)
  ax[3].plot(t, fGP, 'b.', alpha = 0.3)
  ax[4].plot(t, fTem, 'b.', alpha = 0.3)
  
  f = [fPDC, fPLD, fPLDGP, fGP, fTem]
  
  # Scale y limits
  out, M, MAD = GetOutliers(np.concatenate(f), sig_tol = 5.)
  [axis.set_ylim(M - 5 * MAD, M + 5 * MAD) for axis in ax]
  
  # Plot histograms
  for fn, ah, a, c in zip(f, axh, ax, ['r', 'r', 'r', 'r', 'b']):
    n, bins, patches = ah.hist(fn[~np.isnan(fn)], normed = 1, 
                               orientation = 'horizontal', 
                               color=c, alpha = 0.25,
                               bins = FreedmanDiaconis(fn))
    ylim = a.get_ylim()
    ah.set_ylim(ylim)
    ah.set_xticks([])
    ah.set_yticks([])
    
    # Now fit a gaussian
    mu, sigma = norm.fit(fn, loc = 0)
    ah.plot(mlab.normpdf(bins, mu, sigma), bins, 'k-', linewidth=2)
  
    # Print some statistics
    rss = np.sum(fn[~np.isnan(fn)] ** 2)
    ks0, ks1 = normaltest(fn[~np.isnan(fn)])
    xleft = ah.get_xlim()[0]
    ytop = ah.get_ylim()[1]
    ah.annotate("\n    RSS = %.3e" % rss, (xleft, ytop), ha='left', va='top', fontsize = 9, color = 'k')
    ah.annotate("\n\n    NRM = (%.3e, %.3e)" % (ks0, ks1), (xleft, ytop), ha='left', va='top', fontsize = 9, color = 'k')
    
  # Annotate
  for q in inp.quarters:
  
    # Check if empty
    if not len(data[q]['time']):
      continue
    
    # Last time and central time for this quarter
    lt = data[q]['time'][-1][-1]
    ct = (data[q]['time'][0][0] + data[q]['time'][-1][-1]) / 2.
        
    # Quarter number
    ax[0].annotate(q, (ct, ax[0].get_ylim()[1]), ha='center', va='bottom', fontsize = 24)
    
    for axis in ax:
      axis.axvline(lt, color='k', ls = '--')

  # Labels and titles
  [axis.set_xticklabels([]) for axis in ax[:-1]]
  [axis.margins(0, 0.01) for axis in ax] 
  ax[0].set_title('Kepler PDC Flux', fontsize = 28, fontweight = 'bold', y = 1.1)
  ax[1].set_title('PLD + Polynomial', fontsize = 28, fontweight = 'bold', y = 1.025) 
  ax[2].set_title('GP Only', fontsize = 28, fontweight = 'bold', y = 1.025)  
  ax[3].set_title('PLD then GP', fontsize = 28, fontweight = 'bold', y = 1.025)
  ax[4].set_title('PLD + GP (Templar)', fontsize = 28, fontweight = 'bold', y = 1.025)

  ax[-1].set_xlabel('Time (Days)', fontsize = 24)
  [axis.set_ylabel(r'$\Delta$ Counts', fontsize = 24) for axis in ax]
  
  # Appearance
  for s in ['top', 'bottom', 'left', 'right']:
    [axis.spines[s].set_linewidth(2) for axis in ax]
    [axis.spines[s].set_linewidth(2) for axis in axh]
  [tick.label.set_fontsize(20) for tick in ax[-1].xaxis.get_major_ticks()]
  [tick.label.set_fontsize(18) for axis in ax for tick in axis.yaxis.get_major_ticks()] 
  
  # Colors
  [axis.set_axis_bgcolor((1., 0.975, 0.975)) for axis in ax[:-1]]
  ax[-1].set_axis_bgcolor((0.975, 0.975, 1.0))
  
  # Save and return
  fig.savefig(os.path.join(inp.datadir, str(inp.id), '_plots', 'comparison.png'), bbox_inches = 'tight')
  
  return fig, ax + axh