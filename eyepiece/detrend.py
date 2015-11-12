#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Input, FunctionWrapper, GetData, GitHash
from .preprocess import Preprocess
from .linalg import LnLike, Whiten, PLDFlux
from .download import DownloadInfo
from scipy.optimize import fmin_l_bfgs_b
import itertools
import numpy as np
import os
import pysyzygy as ps

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

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
  for time, fpix, perr in zip(dq['time'], dq['fpix'], dq['perr']):
    res = LnLike(x, time, fpix, perr, kernel = kernel)
    ll += res[0]
    grad_ll += res[1]
    
  return (-ll, -grad_ll)

def QuarterDetrend(tag, id, kernel, kinit, sigma, kbounds, maxfun, debug, datadir):
  '''
  
  '''

  # Debug?
  if debug:
    print("Begin tag " + str(tag))

  # Load the data (if necessary)
  global data
  if data is None:
    data = GetData(id, data_type = 'bkg', datadir = datadir)
  detpath = os.path.join(datadir, str(id), '_detrend')
  
  # Tags: i is the iteration number; q is the quarter number
  i = tag[0]
  q = tag[1]
  qd = data[q]

  # Is there data for this quarter?
  if qd['time'] == []:
    return None

  # Set our initial guess
  npix = qd['fpix'][0].shape[1]
  init = np.append(kinit, [np.median(np.sum(qd['fpix'][0], axis = 1))] * npix)
  bounds = np.concatenate([kbounds, [[-np.inf, np.inf]] * npix])

  # Perturb initial conditions by sigma, and ensure within bounds
  np.random.seed(tag)
  foo = bounds[:,0]
  while np.any(foo <= bounds[:,0]) or np.any(foo >= bounds[:,1]):
    foo = init * (1 + sigma * np.random.randn(len(init)))
  init = foo
  
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

  for t, fp, pe in zip(qd['time'], qd['fpix'], qd['perr']):
    res = LnLike(x, t, fp, pe, kernel = kernel, predict = True)
    time = np.append(time, t)
    fsum = np.append(fsum, np.sum(fp, axis = 1))
    ypld = np.append(ypld, res[2])
    gpmu = np.append(gpmu, res[3])
    gperr = np.append(gperr, res[4])

  res = {'time': time, 'fsum': fsum, 'ypld': ypld, 'gpmu': gpmu, 'gperr': gperr,
          'x': x, 'lnlike': lnlike, 'info': info, 'init': init, 'tag': tag}

  # Save
  np.savez(os.path.join(detpath, '%02d.%02d.npz' % res['tag'][::-1]), **res)

  # Debug?
  if debug:
    print("End tag " + str(tag))

  return res

def Detrend(input_file = None, pool = None):

  '''

  '''
  
  # Reset the global variable
  global data
  data = None
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  if not os.path.exists(detpath):
    os.makedirs(detpath)
  kernel = inp.kernel
  iPLD = len(inp.kernel.pars)
  
  # Run inspection if needed
  success = Preprocess(input_file)
  if not success:
    if not inp.quiet:
      print("Detrending aborted!")
    return False
  
  # We're going to update the data with the detrending info later
  tdata = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  pdata = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  
  # Have we already detrended?
  if not inp.clobber:
    try:
      for q in pdata:
        if len(pdata[q]['time']) == 0:
          continue
        if pdata[q]['dvec'] is None:
          raise ValueError("Detrending vector is ``None``.")
      
      # Data is already detrended
      if not inp.quiet:
        print("Using existing detrending info.")
      return True
      
    except ValueError:
      
      # We're going to have to detrend
      pass

  if not inp.quiet:
    print("Detrending...")

  # Set up our list of runs
  if not inp.clobber:
    quarters = []
    for q in inp.quarters:
      if tdata[q]['dvec'] is None:
        quarters.append(q)
  else:
    quarters = inp.quarters
  
  tags = list(itertools.product(range(inp.niter), quarters))
  FW = FunctionWrapper(QuarterDetrend, inp.id, inp.kernel, inp.kinit, inp.pert_sigma, 
                       inp.kbounds, inp.maxfun, inp.debug, inp.datadir)
  
  # Parallelize?
  if pool is None:
    M = map
  else:
    M = pool.map
  
  # Run
  for res in M(FW, tags):
    continue
  
  # Identify the highest likelihood run
  for q in quarters:
    files = [os.path.join(detpath, f) for f in os.listdir(detpath) 
             if f.startswith('%02d.' % q) and f.endswith('.npz')]
    
    # Is there data this quarter?
    if len(files) == 0:
      continue

    # Grab the highest likelihood run and update the ``trn.npz``
    lnl = np.zeros_like(files, dtype = float)
    for i, f in enumerate(files):
      lnl[i] = float(np.load(f)['lnlike'])
    res = np.load(files[np.argmax(lnl)])
    tdata[q].update({'dvec': res['x']})
    pdata[q].update({'dvec': res['x']})
    
    tdata[q].update({'info': res['info'][()]})
    pdata[q].update({'info': res['info'][()]})
    
    tdata[q].update({'lnlike': res['lnlike']})
    pdata[q].update({'lnlike': res['lnlike']})
    
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'trn.npz'), data = tdata, hash = GitHash())  
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'prc.npz'), data = pdata, hash = GitHash())  

def ComputePLD(input_file = None):
  '''

  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  iPLD = len(inp.kernel.pars)
  
  # Have we already detrended?
  pdata = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  tdata = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  if not inp.clobber:
    try:
      for q in pdata:
        if len(pdata[q]['time']) == 0:
          continue
        if len(pdata[q]['pmod']) == 0:
          raise ValueError("Pixel model list is empty.")
      
      # Data is already detrended
      if not inp.quiet:
        print("Using existing PLD info.")
      return True
      
    except ValueError:
      
      # We're going to have to compute
      pass
  
  # Get some info
  info = DownloadInfo(inp.id, inp.dataset, trninfo = inp.trninfo, 
                      inject = inp.inject, datadir = inp.datadir)
  per = info['per']
  rhos = info['rhos']
  tN = info['tN']
  
  # Get whitened transits
  if len(tN):
  
    # We're going to whiten the data and fit a simple transit model so we 
    # can go back and derive the correct PLD errors in-transit.
    if not inp.clobber:
      try:
        foo = np.load(os.path.join(inp.datadir, str(inp.id), '_data', 'white.npz'))
        t = foo['t']
        f = foo['f']
        e = foo['e']
      except FileNotFoundError:  
        t, f, e = GetWhitenedData(input_file = input_file, folded = True, return_mean = True)
        np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'white.npz'), t = t, f = f, e = e)
    else:
      t, f, e = GetWhitenedData(input_file = input_file, folded = True, return_mean = True)
      np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'white.npz'), t = t, f = f, e = e)
  
    if not inp.quiet:
      print("Computing approximate transit model...")

    # We're just going to solve for RpRs, bcirc, q1 and q2 here
    def chisq(x):
      '''
    
      '''
    
      RpRs, bcirc, q1, q2 = x
      psm = ps.Transit(per = per, q1 = q1, q2 = q2, RpRs = RpRs, rhos = rhos, 
                       t0 = 0., ecw = 0., esw = 0., bcirc = bcirc, MpMs = 0.)
      tmod = psm(t, 'binned')
      c = np.sum( ((tmod - f) / e) ** 2 )

      return c
  
    # Run the optimizer
    init = [info['RpRs'], info['b'], 0.25, 0.25]
    bounds = [[1.e-4, 0.5], [0., 1.], [0., 1.], [0., 1.]]
    res = fmin_l_bfgs_b(chisq, init, approx_grad = True, bounds = bounds)
    RpRs, bcirc, q1, q2 = res[0]
  
    # Save our quick-and-dirty transit fit
    psm = ps.Transit(per = per, q1 = q1, q2 = q2, RpRs = RpRs, rhos = rhos, 
                     t0 = 0., ecw = 0., esw = 0., bcirc = bcirc, MpMs = 0.)
    tmod = psm(t, 'binned')
    np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'tmod.npz'), t = t, tmod = tmod)

    # Pre-compute the transit model, now on the actual (unfolded) time series
    psm = ps.Transit(per = per, q1 = q1, q2 = q2, RpRs = RpRs, rhos = rhos, 
                     tN = tN, ecw = 0., esw = 0., bcirc = bcirc, MpMs = 0.)
  
  # Now, finally, compute the PLD flux and errors
  # First, reload the data
  tdata = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  pdata = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  
  for q in inp.quarters:
  
    if len(pdata[q]['time']) == 0:
      continue
    
    # PLD and GP coeffs for this quarter
    c = pdata[q]['dvec'][iPLD:]
    x = pdata[q]['dvec'][:iPLD]
    
    # Crowding parameter
    crwd = pdata[q]['crwd']
    
    # Reset
    tdata[q]['pmod'] = []
    tdata[q]['yerr'] = []
    tdata[q]['ypld'] = []
    
    pdata[q]['pmod'] = []
    pdata[q]['yerr'] = []
    pdata[q]['ypld'] = []
    
    # Loop over all transits
    for time, fpix, perr in zip(tdata[q]['time'], tdata[q]['fpix'], tdata[q]['perr']):
      
      # Compute the transit model
      if len(tN):
        tmod = psm(time, 'binned')
      else:
        tmod = 1.
      
      # Compute the PLD model
      pmod, ypld, yerr = PLDFlux(c, fpix, perr, tmod, crowding = crwd)
      
      # The pixel model
      tdata[q]['pmod'].append(pmod)
      
      # The errors on our PLD-detrended flux
      tdata[q]['yerr'].append(yerr)
      
      # The PLD-detrended, transitless flux
      # NOTE: This is just for verification purposes, since we used
      #       a very quick transit optimization to compute this!
      tdata[q]['ypld'].append(ypld)
  
    # Now loop over all chunks in the full (processed) data
    for time, fpix, perr in zip(pdata[q]['time'], pdata[q]['fpix'], pdata[q]['perr']):
      
      # Compute the transit model
      if len(tN):
        tmod = psm(time, 'binned')
      else:
        tmod = 1.
      
      # Compute the PLD model
      pmod, ypld, yerr = PLDFlux(c, fpix, perr, tmod, crowding = crwd)
      
      # The pixel model
      pdata[q]['pmod'].append(pmod)
      
      # The errors on our PLD-detrended flux
      pdata[q]['yerr'].append(yerr)
      
      # The PLD-detrended, transitless flux
      # NOTE: This is just for verification purposes, since we used
      #       a very quick transit optimization to compute this!
      pdata[q]['ypld'].append(ypld)
  
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'trn.npz'), data = tdata, hash = GitHash())
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'prc.npz'), data = pdata, hash = GitHash())
  
  return True

def GetWhitenedData(input_file = None, folded = True, return_mean = False):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  
  if not inp.quiet:
    print("Whitening the flux...")
    
  # Load the data
  bkg = GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
  prc = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  info = DownloadInfo(inp.id, inp.dataset, trninfo = inp.trninfo, 
                      inject = inp.inject, datadir = inp.datadir)
  tN = info['tN']
  per = info['per']

  if folded and len(tN) == 0:
    raise Exception("No transits for current target!")

  # Whiten
  t = np.array([], dtype = float)
  f = np.array([], dtype = float)
  e = np.array([], dtype = float)
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
    x = np.load(files[np.argmax(lnl)])['x']
    
    for b_time, b_fpix, b_perr, time, fpix, perr in zip(bkg[q]['time'], bkg[q]['fpix'], bkg[q]['perr'], prc[q]['time'], prc[q]['fpix'], prc[q]['perr']):
    
      # Whiten the flux
      flux, ferr = Whiten(x, b_time, b_fpix, b_perr, time, fpix, perr, kernel = inp.kernel, crowding = prc[q]['crwd'], return_mean = return_mean)
    
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

def GetBadChunks(f, winsz = 50, sig_tol = 3., sort = False):
  '''
  
  '''
  
  tmp = np.zeros_like(f) * np.nan
  rss = np.zeros_like(f) * np.nan

  for i in range(winsz, len(f) - winsz):

    # Grab the raw sum of the squares of the residuals
    fwin = f[i - winsz / 2 : i + winsz / 2]
    tmp[i] = np.sum(fwin ** 2)
  
    # Remove the point that's farthest from the mean
    # and recalculate; this prevents single outliers
    # from spoiling a whole chunk of data
    fwin = np.delete(fwin, np.argmax(np.abs(fwin)))
    rss[i] = np.sum(fwin ** 2)

  # Find the "bad" chunks of data -- those with the
  # highest RSS (residual sum of squares) values
  M = np.nanmedian(tmp)
  MAD = 1.4826 * np.nanmedian(np.abs(tmp - M))
  bad = []
  for i, x in enumerate(rss):
    if (x > M + sig_tol * MAD):
      bad.extend(list(range(int(i - winsz / 2), int(i + winsz / 2))))  
  bad = np.array(sorted(list(set(bad))))

  # We now have all the indices of the "bad" data points
  # Let's group these into discrete "chunks"
  i = [-1] + list(np.where(bad[1:] - bad[:-1] > 1)[0]) + [len(bad) - 1]
  chunks = [bad[a+1:b+1] for a,b in zip(i[:-1], i[1:])]

  if sort:
    # Finally, let's sort the chunks in (decreasing) order of RSS,
    # so that the "worst" chunks are the first ones in the list
    chunks = [c for (r,c) in sorted(zip([-np.sum(f[chunk] ** 2) for chunk in chunks], chunks))]

  # Return the indices of the "bad" chunks
  return chunks