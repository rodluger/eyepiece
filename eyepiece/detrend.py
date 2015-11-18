#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Input, FunctionWrapper, GetData, GitHash
from .preprocess import Preprocess
from .linalg import LnLike, PLDFlux
from .download import DownloadInfo
from scipy.optimize import fmin_l_bfgs_b, curve_fit
import itertools
import numpy as np
import os
import pysyzygy as ps
import george
import random

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

data = None

def NegLnLike(x, id, q, kernel, debug, maskpix):
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
    res = LnLike(x, time, fpix, perr, kernel = kernel, maskpix = maskpix)
    ll += res[0]
    grad_ll += res[1]
    
  return (-ll, -grad_ll)

def QuarterDetrend(tag, id, kernel, kinit, sigma, kbounds, maxfun, debug, datadir, pld_guess, maxpix):
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
  iPLD = len(kinit)
  
  # Tags: i is the iteration number; q is the quarter number
  i = tag[0]
  q = tag[1]
  qd = data[q]

  # Is there data for this quarter?
  if qd['time'] == []:
    return None

  # Set our initial guess and bounds
  npix = qd['fpix'][0].shape[1]
  
  if pld_guess == 'random':
    pld_guess = random.choice(['constant', 'linear'])
  if pld_guess == 'constant':
    # Constant (simple) initial guess
    init = [np.median(np.sum(qd['fpix'][0], axis = 1))] * npix
  elif pld_guess == 'linear':
    # Solve the (linear) PLD problem for this quarter
    t_all = np.array([], dtype = float)
    y_all = np.array([], dtype = float)
    fpix = np.array([x for y in qd['fpix'] for x in y])
    fsum = np.sum(fpix, axis = 1)
    p0 = np.array([np.median(fsum)] * fpix.shape[1])
    def pm(y, *x):
      return np.sum(fpix * np.outer(1. / fsum, x), axis = 1)
    init, _ = curve_fit(pm, None, fsum, p0 = p0)    
  else:
    raise ValueError('Invalid setting for ``pld_guess``.')
    
  init = np.append(kinit, init)
  bounds = np.concatenate([kbounds, [[-np.inf, np.inf]] * npix])

  # Perturb initial conditions by sigma, and ensure within bounds
  np.random.seed(tag)
  foo = bounds[:,0]
  while np.any(foo <= bounds[:,0]) or np.any(foo >= bounds[:,1]):
    foo = init * (1 + sigma * np.random.randn(len(init)))
  init = foo
  
  # Mask the faintest pixels?
  if maxpix:
    # Get the pixel indices, sorted from brightest to faintest
    idx = np.argsort(-np.median([x for y in qd['fpix'] for x in y], axis = 0))
    if len(idx) > maxpix:
      maskpix = idx[maxpix:]
  else:
    maskpix = []
  
  # Run the optimizer.
  res = fmin_l_bfgs_b(NegLnLike, init, approx_grad = False,
                      args = (id, q, kernel, debug, maskpix), bounds = bounds,
                      m = 10, factr = 1.e1, pgtol = 1e-05, maxfun = maxfun)  

  # Grab some info
  x = res[0]
  lnlike = -res[1]
  info = res[2]       
  
  # Mask the faintest pixels?
  if maxpix:
    x[iPLD:][maskpix] = 0

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

def Detrend(input_file = None, pool = None, clobber = False):

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
  if not (inp.clobber or clobber):
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
  if not (inp.clobber or clobber):
    quarters = []
    for q in inp.quarters:
      if tdata[q]['dvec'] is None:
        quarters.append(q)
  else:
    quarters = inp.quarters
  
  tags = list(itertools.product(range(inp.niter), quarters))
  FW = FunctionWrapper(QuarterDetrend, inp.id, inp.kernel, inp.kinit, inp.pert_sigma, 
                       inp.kbounds, inp.maxfun, inp.debug, inp.datadir, inp.pld_guess,
                       inp.maxpix)
  
  # Parallelize?
  if pool is None:
    M = map
  else:
    try:
      # Let's try to do this asyncronously
      M = pool.imap_unordered
    except:
      M = pool.map
  
  # Run
  njobs = len(tags)
  n = 0
  for res in M(FW, tags):
    n += 1
    if not inp.quiet:
      print("Completed task %d/%d." % (n, njobs))
  
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

def ComputePLD(input_file = None, clobber = False):
  '''

  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  iPLD = len(inp.kernel.pars)
  
  # Have we already detrended?
  pdata = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  tdata = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  
  if not (inp.clobber or clobber):
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
                      inject = inp.inject, datadir = inp.datadir,
                      clobber = False, ttvs = inp.ttvs,
                      pad = inp.padbkg)
  per = info['per']
  rhos = info['rhos']
  tN = info['tN']
  
  # Optimize transit model
  if len(tN):
  
    if not inp.quiet:
      print("Computing approximate transit model...")
    
    # This lets us approximately solve for RpRs, bcirc, q1, q2
    def negll(x):
      '''
      
      '''
      
      RpRs, bcirc, q1, q2 = x
      psm = ps.Transit(per = per, q1 = q1, q2 = q2, RpRs = RpRs, rhos = rhos, 
                       tN = tN, ecw = 0., esw = 0., bcirc = bcirc, MpMs = 0.)
      ll = 0
      
      # Loop over all quarters                 
      for q in inp.quarters:
  
        # Empty?
        if len(pdata[q]['time']) == 0:
          continue
    
        # Info for this quarter
        crwd = tdata[q]['crwd']
        c = tdata[q]['dvec'][iPLD:]
        x = tdata[q]['dvec'][:iPLD]
        inp.kernel.pars = x
        gp = george.GP(inp.kernel)
        
        # Loop over all transits
        for time, fpix, perr in zip(tdata[q]['time'], tdata[q]['fpix'], tdata[q]['perr']):
      
          # Compute the transit model
          try:
            tmod = psm(time, 'binned')
          except Exception as e: 
            if inp.debug:
              print("Transit optimization exception:", str(e))  
            return 1.e20
      
          # Compute the PLD model
          pmod, ypld, yerr = PLDFlux(c, fpix, perr, tmod, crowding = crwd)
      
          # Compute the GP model
          try:
            gp.compute(time, yerr)
          except Exception as e: 
            if inp.debug:
              print("Transit optimization exception:", str(e))
            return 1.e20
            
          ll += gp.lnlikelihood(ypld)
      
      if inp.debug:
        print("Likelihood: ", ll)
      
      return -ll
    
    # Run the optimizer.
    init = [info['RpRs'], info['b'], 0.25, 0.25]
    bounds = [[1.e-4, 0.5], [0., 0.95], [0., 1.], [0., 1.]]
    res = fmin_l_bfgs_b(negll, init, approx_grad = True, bounds = bounds)
    RpRs, bcirc, q1, q2 = res[0]
    
    # Save these!
    np.savez(os.path.join(inp.datadir, str(inp.id), '_data', 'rbqq.npz'), RpRs = RpRs, bcirc = bcirc, q1 = q1, q2 = q2)
    
    # Pre-compute the transit model
    psm = ps.Transit(per = per, q1 = q1, q2 = q2, RpRs = RpRs, rhos = rhos, 
                     tN = tN, ecw = 0., esw = 0., bcirc = bcirc, MpMs = 0.)
  
  # Now, finally, compute the PLD flux and errors  
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

def GetBadChunks(f, winsz = 50, sig_tol = 3., maxsz = 300, sort = False):
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
  
  # Delete chunks that are too big. These are likely due to 
  # entire quarters having larger RMS, and we're not interested in those.
  chunks = [c for c in chunks if len(c) < maxsz]

  return chunks