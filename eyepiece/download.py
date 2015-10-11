#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
download.py
-----------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import eyepiece.config as config
from .utils import GitHash
import os
import kplr
import numpy as np

__all__ = ['GetData', 'GetInfo']

def EmptyData(quarters):
  '''
  
  '''
  
  foo = {}
  for q in quarters:
    foo.update({q: {'time': [], 'fsum': [], 'ferr': [], 'fpix': [], 
                   'perr': [], 'cad': []}})
  return foo

def GetKoi(koi):
  '''
  A wrapper around :py:func:`kplr.API().koi`, with the additional command
  `select=*` to query **all** columns in the Exoplanet database.

  :param float koi_number: The full KOI number of the target (`XX.XX`)

  :returns: A :py:mod:`kplr` `koi` object

  '''
  client = kplr.API()
  kois = client.kois(where="kepoi_name+like+'K{0:08.2f}'"
                     .format(float(koi)), select="*")
  if not len(kois):
    raise ValueError("No KOI found with the number: '{0}'".format(koi))
  return kois[0]

def GetTransitTimes(koi, tstart, tend, pad = 2.0, ttvs = False, long_cadence = True):
  '''
  
  '''

  planet = GetKoi(koi)
  per = planet.koi_period
  t0 = planet.koi_time0bk
  tdur = planet.koi_duration/24.
  
  if ttvs:
    # Get transit times (courtesy Ethan Kruse)
    try:
      with open(os.path.join(config.ttvpath, "KOI%.2f.ttv" % koi), "r") as f:
        lines = [l for l in f.readlines() if ('#' not in l) and (len(l) > 1)]
        tcads = np.array([int(l.split('\t')[1]) for l in lines], dtype = 'int32')
    except IOError:
      raise Exception("Unable to locate TTV file for the target.")
  
    # Ensure t0 is in fact the first transit
    t0 -= per * divmod(t0 - tstart, per)[0]
  
    # Calculate transit times from cadences
    if long_cadence:
      tN = t0 + (tcads - tcads[0]) * KEPLONGEXP
    else:
      tN = t0 + (tcads - tcads[0]) * KEPSHRTEXP

    # Ensure our transit times go all the way to the end, plus a bit
    # (the extra per/2 is VERY generous; we're likely fitting for one
    # or two more transits than we have data for.)
    while tN[-1] < tend + per/2.:
      tN = np.append(tN, tN[-1] + per)
  
  else:
    n, r = divmod(tstart - t0, per)
    if r < (pad * tdur)/2.: t0 = t0 + n*per
    else: t0 = t0 + (n + 1)*per
    tN = np.arange(t0, tend + per, per)

  return tN, per, tdur

def GetTPFData(koi, long_cadence = True, clobber = False, 
               bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17], pad = 2.0,
               aperture = 'optimal', quarters = range(18), datadir = config.datadir,
               ttvs = False, quiet = False):
  '''
  
  '''
  
  if not clobber:
    try:
      data = np.load(os.path.join(datadir, str(koi), 'data_raw.npz'))['data'][()]     # For some reason, numpy saved it as a 0-d array! See http://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
      foo = np.load(os.path.join(datadir, str(koi), 'transits.npz'))
      tN = foo['tN']
      per = foo['per']
      tdur = foo['tdur']
      if not quiet: print("Loading saved TPF data...")
      return data, tN, per, tdur
    except:
      pass
  
  if not quiet: print("Downloading TPF data...")
  planet = GetKoi(koi)  
  data = EmptyData(quarters)
  tpf = planet.get_target_pixel_files(short_cadence = not long_cadence)
  if len(tpf) == 0:
    raise Exception("No pixel files for selected object!")
  tstart = np.inf
  tend = -np.inf
  for fnum in range(len(tpf)):
    with tpf[fnum].open(clobber = clobber) as f:  
      ap = f[2].data
      if aperture == 'optimal':
        idx = np.where(ap & 2)
      elif aperture == 'full':
        idx = np.where(ap & 1)
      else:
        raise Exception('ERROR: Invalid aperture setting `%s`.' % aperture)
      qdata = f[1].data
      quarter = f[0].header['QUARTER']
      crowding = f[1].header['CROWDSAP']
    time = np.array(qdata.field('TIME'), dtype='float64')
    nan_inds = list(np.where(np.isnan(time))[0])                                      # Any NaNs will screw up the transit model calculation,
    time = np.delete(time, nan_inds)                                                  # so we need to remove them now.
    cad = np.array(qdata.field('CADENCENO'), dtype='int32')
    cad = np.delete(cad, nan_inds)
    flux = np.array(qdata.field('FLUX'), dtype='float64')
    flux = np.delete(flux, nan_inds, 0)
    fpix = np.array([f[idx] for f in flux], dtype='float64')
    perr = np.array([f[idx] for f in qdata.field('FLUX_ERR')], dtype='float64')
    perr = np.delete(perr, nan_inds, 0)
    fsum = np.sum(fpix, axis = 1)
    ferr = np.sum(perr**2, axis = 1)**0.5
    quality = qdata.field('QUALITY')
    quality = np.delete(quality, nan_inds)
    qual_inds = []
    for b in bad_bits:
      qual_inds += list(np.where(quality & 2**(b-1))[0])
    nan_inds = list(np.where(np.isnan(fsum))[0])
    bad_inds = np.array(sorted(list(set(qual_inds + nan_inds))))
    time = np.delete(time, bad_inds)
    cad = np.delete(cad, bad_inds)
    fsum = np.delete(fsum, bad_inds)
    ferr = np.delete(ferr, bad_inds)
    fpix = np.delete(fpix, bad_inds, 0)
    perr = np.delete(perr, bad_inds, 0)
    data[quarter].update({'time': time, 'fsum': fsum, 'ferr': ferr, 'fpix': fpix, 
                          'perr': perr, 'cad': cad, 'crowding': crowding})
    if time[0] < tstart: tstart = time[0]
    if time[-1] > tend: tend = time[-1]
  
  if not os.path.exists(os.path.join(datadir, str(koi))):
    os.makedirs(os.path.join(datadir, str(koi)))
  np.savez_compressed(os.path.join(datadir, str(koi), 'data_raw.npz'), data = data, hash = GitHash())
  
  # Now get the transit info
  tN, per, tdur = GetTransitTimes(koi, tstart, tend, pad = pad, ttvs = ttvs, 
                                  long_cadence = long_cadence)
  np.savez_compressed(os.path.join(datadir, str(koi), 'transits.npz'), tN = tN, per = per, tdur = tdur, hash = GitHash())
    
  return data, tN, per, tdur

def GetData(koi, data_type = 'proc', blind = False, datadir = config.datadir):
  '''
  
  '''

  try:
    data = np.load(os.path.join(datadir, str(koi), 'data_%s.npz' % data_type))['data'][()]
  except IOError:
    Inspect(koi = koi, blind = blind)
    data = np.load(os.path.join(datadir, str(koi), 'data_%s.npz' % data_type))['data'][()]

  return data

def GetInfo(koi, datadir = config.datadir):
  '''
  
  '''
  
  try:
    tN = np.load(os.path.join(datadir, str(koi), 'transits.npz'))['tN'][()]
    per = np.load(os.path.join(datadir, str(koi), 'transits.npz'))['per'][()]
    tdur = np.load(os.path.join(datadir, str(koi), 'transits.npz'))['tdur'][()]
    hash = np.load(os.path.join(datadir, str(koi), 'transits.npz'))['hash'][()]
  except IOError:
    raise Exception("File ``transits.npz`` not found.")
  
  return tN, per, tdur, hash