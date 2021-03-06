#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
download.py
-----------

.. todo::
   - Add TTV support; use Rowe+15 posteriors as input.

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import GitHash
import kplr
import os
import numpy as np
import pysyzygy as ps

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

def get_kplr_obj(id):
  '''
  A wrapper around :py:func:`kplr.API().koi` and :py:func:`kplr.API().star`, 
  with the additional command `select=*` to query **all** columns in the 
  Exoplanet database.

  :param float id: The full KOI (`XX.XX`) or KIC (`XXX...`) number of the target

  :returns: A :py:mod:`kplr` ``koi`` object

  '''
  client = kplr.API()
  
  if type(id) is float:
    kois = client.kois(where="kepoi_name+like+'K{0:08.2f}'"
                       .format(float(id)), select="*")
    if not len(kois):
      raise ValueError("No KOI found with the number: '{0}'".format(id))
    return kois[0]
  elif type(id) is int:
    try:
      star = kplr.API().star(id)
    except:
      raise ValueError("No KIC found with the number: '{0}'".format(id))
    return star
  else:
    raise ValueError("Unrecognized id: '{0}'".format(id))

def EmptyData(quarters):
  '''
  
  '''
  
  foo = {}
  
  for q in quarters:
    foo.update({q: {'time': [], 'cadn': [], 'fpix': [], 'perr': [], 
                    'pdcf': [], 'pdce': [], 'crwd': None, 'dvec': None,
                    'pmod': [], 'yerr': [], 'trni': [], 'info': None,
                    'lnlike': None, 'ypld': []}})
  return foo

def DownloadKeplerData(id, datadir = '', long_cadence = True, clobber = False, 
                  bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17],
                  aperture = 'optimal', quarters = range(18), quiet = False,
                  inject = {}, ttvs = False, pad = 2.0, trninfo = {}, **kwargs):
  '''
  
  '''
  
  if not clobber:
  
    try:
    
      # For some reason, numpy saves it as a 0-d array.
      # See http://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
    
      data = np.load(os.path.join(datadir, str(id), '_data', 'raw.npz'))['data'][()]
      
      if not quiet: 
        print("Loading saved TPF data...")
      
      return data
    
    except FileNotFoundError:
      pass
  
  # Create directories
  for sd in ['_data', '_plots', '_detrend']:
    if not os.path.exists(os.path.join(datadir, str(id), sd)):
      os.makedirs(os.path.join(datadir, str(id), sd))
  
  if not quiet: 
    print("Downloading TPF data...")
  
  # Download the data
  try:
    obj = get_kplr_obj(id)
  except:
    raise Exception("Unable to access online database.")
  data = EmptyData(quarters)
  
  # Target pixel files
  tpf = obj.get_target_pixel_files(short_cadence = not long_cadence)
  
  # Lightcurves
  lcs = obj.get_light_curves(short_cadence = not long_cadence)
  
  nfiles = len(tpf)
  
  if nfiles == 0:
    raise Exception("No pixel files for selected object!")
  elif len(lcs) != nfiles:
  
    # This shouldn't happen. If it does, I'll need to re-code this section
    raise Exception("Mismatch in number of lightcurves and number of pixel files.")

  # Loop over the target pixel files/lightcurve files
  for fnum in range(nfiles):
    with tpf[fnum].open(clobber = clobber) as f:  
      ap = f[2].data
      qdata = f[1].data
      quarter = f[0].header['QUARTER']
      if quarter not in quarters:
        continue
      crwd = f[1].header['CROWDSAP']
    
    with lcs[fnum].open(clobber = clobber) as f:
      hdu_data = f[1].data
      pdcf = np.array(hdu_data["pdcsap_flux"] - np.nanmedian(hdu_data["pdcsap_flux"]))
      pdce = np.array(hdu_data["pdcsap_flux_err"])
    
    if aperture == 'optimal':
      idx = np.where(ap & 2)
    elif aperture == 'full':
      idx = np.where(ap & 1)
    else:
      raise Exception('ERROR: Invalid aperture setting `%s`.' % aperture)
    time = np.array(qdata.field('TIME'), dtype='float64')
    
    if len(time) != len(pdcf):
      
      # This shouldn't happen. If it does, I'll need to re-code this section
      raise Exception('Mismatch in length of pixel flux and PDC flux!')
    
    # Any NaNs will screw up the transit model calculation,
    # so we need to remove them now.
    nan_inds = list(np.where(np.isnan(time))[0])                                      
    time = np.delete(time, nan_inds)                                                  
    cadn = np.array(qdata.field('CADENCENO'), dtype='int32')
    cadn = np.delete(cadn, nan_inds)
    flux = np.array(qdata.field('FLUX'), dtype='float64')
    flux = np.delete(flux, nan_inds, 0)
    fpix = np.array([f[idx] for f in flux], dtype='float64')
    
    # Inject transits?
    if (inject is not None) and (inject != {}) and (inject != False):
      psm = ps.Transit(**inject)
      tmod = (psm(time, 'binned') - 1.) * crwd + 1.
      for n in range(fpix.shape[1]):
        fpix[:, n] *= tmod 
    
    perr = np.array([f[idx] for f in qdata.field('FLUX_ERR')], dtype='float64')
    perr = np.delete(perr, nan_inds, 0)
    pdcf = np.delete(pdcf, nan_inds, 0)
    pdce = np.delete(pdce, nan_inds, 0)
    
    # Remove bad bits
    quality = qdata.field('QUALITY')
    quality = np.delete(quality, nan_inds)
    qual_inds = []
    for b in bad_bits:
      qual_inds += list(np.where(quality & 2 ** (b - 1))[0])
    nan_inds = list(np.where(np.isnan(np.sum(fpix, axis = 1)))[0])
    bad_inds = np.array(sorted(list(set(qual_inds + nan_inds))))
    time = np.delete(time, bad_inds)
    cadn = np.delete(cadn, bad_inds)
    fpix = np.delete(fpix, bad_inds, 0)
    perr = np.delete(perr, bad_inds, 0)
    pdcf = np.delete(pdcf, bad_inds, 0)
    pdce = np.delete(pdce, bad_inds, 0)
    
    data[quarter].update({'time': time, 'fpix': fpix, 'perr': perr, 'cadn': cadn, 'crwd': crwd, 'pdcf': pdcf, 'pdce': pdce})

  # Save to disk
  np.savez_compressed(os.path.join(datadir, str(id), '_data', 'raw.npz'), data = data, hash = GitHash())
  
  # Download the info so we have it saved on disk
  DownloadKeplerInfo(id, datadir = datadir, clobber = clobber, ttvs = ttvs, pad = pad, inject = inject, trninfo = trninfo)
      
  return data

def DownloadKeplerInfo(id, datadir = '', clobber = False, ttvs = False, pad = 2.0, inject = {}, trninfo = {}):
  '''
  
  '''
  
  if not clobber:
    try:
      data = np.load(os.path.join(datadir, str(id), '_data', 'info.npz'))      
      tN = data['tN'][()]
      per = data['per'][()]
      tdur = data['tdur'][()]
      hash = data['hash'][()]
      rhos = data['rhos'][()]
      b = data['b'][()]
      RpRs = data['RpRs'][()]
      foo = {'tN': tN, 'per': per, 'tdur': tdur, 'hash': hash, 'rhos': rhos, 'b': b, 'RpRs': RpRs}
      foo.update(trninfo)
      return foo
    except FileNotFoundError:
      pass
  
  # Grab first and last timestamps for this planet
  data = np.load(os.path.join(datadir, str(id), '_data', 'raw.npz'))['data'][()]
  tstart = np.inf
  tend = -np.inf
  for q in data:
    if len(data[q]['time']) == 0:
      continue
    if data[q]['time'][0] < tstart:
      tstart = data[q]['time'][0]
    if data[q]['time'][-1] > tend:
      tend = data[q]['time'][-1]
  
  if type(id) is float:

    # Grab info from the database
    planet = get_kplr_obj(id)
    per = planet.koi_period
    t0 = planet.koi_time0bk
    tdur = planet.koi_duration/24.
    rhos = planet.koi_srho
    RpRs = planet.koi_ror
    b = planet.koi_impact
  
    # Get the transit times
    if not ttvs:
      n, r = divmod(tstart - t0, per)
      if r < (pad * tdur)/2.: 
        t0 = t0 + n*per
      else: 
        t0 = t0 + (n + 1)*per
      tN = np.arange(t0, tend + per, per)
    else:
      raise Exception("TTV support not yet implemented.")
  
  elif (inject is not None) and (inject != {}) and (inject != False):
    
    # No error handling here -- all params better be present!
    
    per = inject['per']
    rhos = inject['rhos']
    b = inject.get('b', None)
    if b is None: b = inject['bcirc']
    RpRs = inject['RpRs']
    
    # Compute transit duration (includes eccentricity correction)
    ecw = inject['ecw']
    esw = inject['esw']
    MpMs = inject['MpMs']
    ecc = np.sqrt(ecw**2 + esw**2)
    aRs = ((6.672e-8 * rhos * (1. + MpMs) * 
          (per * 86400.)**2.) / (3. * np.pi))**(1./3.)
    inc = np.arccos(b/aRs)
    becc = b * (1 - ecc**2)/(1 - esw)
    # This is the FULL transit duration (Sara Seager's "Exoplanets" pg 58, eqn. 15 in Winn review)
    tdur = per / np.pi * np.arcsin(((1. + RpRs)**2 - becc**2)**0.5 / (np.sin(inc) * aRs))
    tdur *= np.sqrt(1. - ecc**2.)/(1. - esw)
    
    t0 = inject.get('t0', None)
    if t0 is None:
      tN = inject['tN']
    else:
      n, r = divmod(tstart - t0, per)
      if r < (pad * tdur)/2.: 
        t0 = t0 + n*per
      else: 
        t0 = t0 + (n + 1)*per
      tN = np.arange(t0, tend + per, per)
  
  else:
    tN = np.array([], dtype = float)
    per = 0.
    tdur = 0.
    rhos = 0.
    b = 0.
    RpRs = 0.
  
  hash = GitHash()
  
  # Save
  np.savez_compressed(os.path.join(datadir, str(id), '_data', 'info.npz'), tN = tN, 
                      per = per, tdur = tdur, rhos = rhos, b = b, RpRs = RpRs, hash = hash)

  foo = {'tN': tN, 'per': per, 'tdur': tdur, 'rhos': rhos, 'b': b, 'RpRs': RpRs, 'hash': hash}
  foo.update(trninfo)
  
  return foo

def DownloadData(id, dataset, **kwargs):
  '''
  
  '''
  
  if dataset == 'Kepler':
    return DownloadKeplerData(id, **kwargs)
  else:
    raise ValueError('Dataset `%s` not currently supported.' % dataset)

def DownloadInfo(id, dataset, **kwargs):
  '''
  
  '''
  
  if dataset == 'Kepler':
    return DownloadKeplerInfo(id, **kwargs)
  else:
    raise ValueError('Dataset `%s` not currently supported.' % dataset)