#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
detrend.py
----------

.. todo::
   - Save ``input.py`` file for each target

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Input, FunctionWrapper, GetData, GitHash
from .preprocess import Preprocess
from .linalg import LnLike, Whiten
from .download import DownloadInfo
from scipy.optimize import fmin_l_bfgs_b
import itertools
import numpy as np
import os
import george

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
  
  # Check the progress by printing to the screen
  if debug:
    print(q, ll)
  
  return (-ll, -grad_ll)

def QuarterDetrend(tag, id, kernel, kinit, sigma, kbounds, maxfun, debug, datadir):
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

  return {'time': time, 'fsum': fsum, 'ypld': ypld, 'gpmu': gpmu, 'gperr': gperr,
          'x': x, 'lnlike': lnlike, 'info': info, 'init': init, 'tag': tag}

def Detrend(input_file = None, pool = None):

  '''

  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  kernel = inp.kernel
  iPLD = len(inp.kernel.pars)
  
  # Run inspection if needed
  success = Preprocess(input_file)
  if not success:
    if not inp.quiet:
      print("Detrending aborted!")
    return False
  
  # We're going to update the trn data with the detrending info later
  data = GetData(inp.id, data_type = 'trn', datadir = inp.datadir)
  
  # Have we already detrended?
  if not inp.clobber:
    try:
      for q in data:
        if len(data[q]['time']) == 0:
          continue
        if data[q]['dvec'] is None:
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
      if data[q]['dvec'] is None:
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
  
  # Run and save
  for res in M(FW, tags):
  
    # No data?
    if res is None:
      continue
  
    # Save
    if not os.path.exists(detpath):
      os.makedirs(detpath)
    np.savez(os.path.join(detpath, '%02d.%02d.npz' % res['tag'][::-1]), **res)
  
    # Print
    if not inp.quiet:
      print("Detrending complete for quarter %d, tag %s." % (res['tag'][1], str(res['tag'][0])))
  
  # Identify the highest likelihood run
  for q in quarters:
  
    files = [os.path.join(detpath, f) for f in os.listdir(detpath) 
             if f.startswith('%02d.' % q) and f.endswith('.npz')]
    
    # Is there data this quarter?
    if len(files) == 0:
      return None

    # Grab the highest likelihood run
    lnl = np.zeros_like(files, dtype = float)
    for i, f in enumerate(files):
      lnl[i] = float(np.load(f)['lnlike'])
    res = np.load(files[np.argmax(lnl)])
  
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
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir); info.update(inp.info)
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