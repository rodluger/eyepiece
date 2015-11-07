#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
download.py
-----------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .utils import GitHash
import os
import kplr
import numpy as np

__all__ = ['GetData', 'GetTPFData', 'GetInfo']

# Kepler cadences
KEPLONGEXP =              (1765.5/86400.)
KEPSHRTEXP =              (58.89/86400.)

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

def GetTransitTimes(koi, tstart, tend, pad = 2.0, ttvs = False, long_cadence = True,
                    ttvpath = ''):
  '''
  
  '''

  planet = GetKoi(koi)
  per = planet.koi_period
  t0 = planet.koi_time0bk
  tdur = planet.koi_duration/24.
  
  if ttvs:
    # Get transit times (courtesy Ethan Kruse)
    try:
      with open(os.path.join(ttvpath, "KOI%.2f.ttv" % koi), "r") as f:
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

def GetTPFData(id, long_cadence = True, clobber = False, 
               bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17], pad = 2.0,
               aperture = 'optimal', quarters = range(18), datadir = '',
               ttvpath = '', ttvs = False, quiet = False):
  '''
  
  '''
  
  if not clobber:
    try:
      data = np.load(os.path.join(datadir, str(id), '_data', 'raw.npz'))['data'][()]      # For some reason, numpy saved it as a 0-d array! See http://stackoverflow.com/questions/8361561/recover-dict-from-0-d-numpy-array
      foo = np.load(os.path.join(datadir, str(id), '_data', 'transits.npz'))
      tN = foo['tN']
      per = foo['per']
      tdur = foo['tdur']
      if not quiet: print("Loading saved TPF data...")
      return {'data': data, 'tN': tN, 'per': per, 'tdur': tdur}
    except:
      pass
  
  for sd in ['_data', '_plots', '_detrend']:
    if not os.path.exists(os.path.join(datadir, str(id), sd)):
      os.makedirs(os.path.join(datadir, str(id), sd))
  
  if not quiet: print("Downloading TPF data...")
  if type(id) is float:
    obj = GetKoi(id)  
  elif type(id) is int:
    try:
      obj = kplr.API().star(id)
    except:
      raise Exception("Invalid KIC number!")
  else:
    raise Exception("Invalid identifier type!")
  data = EmptyData(quarters)
  tpf = obj.get_target_pixel_files(short_cadence = not long_cadence)
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
  
  np.savez_compressed(os.path.join(datadir, str(id), '_data', 'raw.npz'), data = data, hash = GitHash())
  
  # Now get the transit info
  if type(id) is float:
    tN, per, tdur = GetTransitTimes(id, tstart, tend, pad = pad, ttvs = ttvs, 
                                    long_cadence = long_cadence, ttvpath = ttvpath)
  else:
    tN = np.array([], dtype = int)
    per = 0.
    tdur = 0.
  np.savez_compressed(os.path.join(datadir, str(id), '_data', 'transits.npz'), tN = tN, 
                      per = per, tdur = tdur, hash = GitHash())
    
  return {'data': data, 'tN': tN, 'per': per, 'tdur': tdur}

def GetData(id, data_type = 'prc', datadir = ''):
  '''
  
  '''

  try:
    data = np.load(os.path.join(datadir, str(id), '_data', '%s.npz' % data_type))['data'][()]
  except IOError:
    raise IOError("File ``data_%s.npz`` not found." % data_type)

  return data

def GetInfo(id, datadir = ''):
  '''
  
  '''
  
  try:
    data = np.load(os.path.join(datadir, str(id), '_data', 'transits.npz'))
    tN = data['tN'][()]
    per = data['per'][()]
    tdur = data['tdur'][()]
    hash = data['hash'][()]
  except IOError:
    raise IOError("File ``transits.npz`` not found.")
  
  return {'tN': tN, 'per': per, 'tdur': tdur, 'hash': hash}

def GetPDCFlux(id, long_cadence = True):
  '''
  
  '''
  
  if type(id) is float:
    obj = GetKoi(id)  
  elif type(id) is int:
    try:
      obj = kplr.API().star(id)
    except:
      raise Exception("Invalid KIC number!")
      
  # Get a list of light curve datasets.
  lcs = obj.get_light_curves(short_cadence = not long_cadence)

  # Loop over the datasets and read in the data.
  time, flux, ferr, quality = [], [], [], []
  for lc in lcs:
      with lc.open() as f:
          # The lightcurve data are in the first FITS HDU.
          hdu_data = f[1].data
          time.extend(hdu_data["time"])
          flux.extend(hdu_data["pdcsap_flux"] - np.nanmedian(hdu_data["pdcsap_flux"]))
          ferr.extend(hdu_data["pdcsap_flux_err"])
          quality.extend(hdu_data["sap_quality"])

  return np.array(time), np.array(flux)