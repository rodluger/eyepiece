#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
eyepiece.py
-----------

Download and visually inspect, split, and correct Kepler lightcurves.

'''

import config
import matplotlib.pyplot as pl
from matplotlib.widgets import RectangleSelector
import numpy as np
import kplr
import os

KEPLONGEXP =              (1765.5/86400.)
KEPSHRTEXP =              (58.89/86400.)

# Disable keyboard shortcuts
pl.rcParams['keymap.all_axes'] = ''
pl.rcParams['keymap.back'] = ''
pl.rcParams['keymap.forward'] = ''
pl.rcParams['keymap.fullscreen'] = ''
pl.rcParams['keymap.grid'] = ''
pl.rcParams['keymap.home'] = ''
pl.rcParams['keymap.pan'] = ''
pl.rcParams['keymap.quit'] = ''
pl.rcParams['keymap.save'] = ''
pl.rcParams['keymap.xscale'] = ''
pl.rcParams['keymap.yscale'] = ''
pl.rcParams['keymap.zoom'] = ''

def EmptyData(quarters):
  '''
  
  '''
  return np.array([{'time': [], 'fsum': [], 'ferr': [], 'fpix': [], 
                    'perr': [], 'cad': []} for q in quarters])

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

def GetTransitTimes(koi, tstart, tend, pad = 4.0, ttvs = False, long_cadence = True):
  '''
  
  '''

  planet = GetKoi(koi)
  per = planet.koi_period
  t0 = planet.koi_time0bk
  tdur = pad * planet.koi_duration/24.
  
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
    if r < tdur/2.: t0 = t0 + n*per
    else: t0 = t0 + (n + 1)*per
    tN = np.arange(t0, tend + per, per)

  return tN, per, tdur

def GetTPFData(koi, long_cadence = True, clobber = False, 
               bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17], pad = 4.0,
               aperture = 'optimal', quarters = range(18), dir = config.dir,
               ttvs = False, quiet = False):
  '''
  
  '''
  
  if not clobber:
    try:
      data = np.load(os.path.join(dir, str(koi), 'data_raw.npz'))['data']
      foo = np.load(os.path.join(dir, str(koi), 'transits.npz'))
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
  
  if not os.path.exists(os.path.join(dir, str(koi))):
    os.makedirs(os.path.join(dir, str(koi)))
  np.savez_compressed(os.path.join(dir, str(koi), 'data_raw.npz'), data = data)
  
  # Now get the transit info
  tN, per, tdur = GetTransitTimes(koi, tstart, tend, pad = pad, ttvs = ttvs, 
                                  long_cadence = long_cadence)
  np.savez_compressed(os.path.join(dir, str(koi), 'transits.npz'), tN = tN, per = per, tdur = tdur)
    
  return data, tN, per, tdur
  
class Selector(object):
  '''
  
  '''
  
  def __init__(self, fig, ax, cad, time, fsum, fpix, tN, tdur, split_cads = [], 
               cad_tol = 10, min_sz = 300, title = ''):
    self.fig = fig
    self.ax = ax
    self.cad = cad
    self.time = time
    self.fsum = fsum
    self.fpix = fpix
    self.tN = tN
    self.tdur = tdur
    self.title = title
    
    # Initialize arrays
    self._outliers = []
    self._transits = []
    self._jumps = []
    self._plots = []
    self.info = ""
    self.state = "fsum"
    
    # Add user-defined split locations
    for s in split_cads:
    
      # What if a split happens in a gap? Shift rightward up to 10 cadences
      for ds in range(10):
        foo = np.where(self.cad == s + ds)[0]
        if len(foo):
          self._jumps.append(foo[0])
          break
    
    # Add cadence gap splits
    cut = np.where(self.cad[1:] - self.cad[:-1] > cad_tol)[0] + 1
    cut = np.concatenate([[0], cut, [len(self.cad) - 1]])                             # Add first and last points
    repeat = True
    while repeat == True:                                                             # If chunks are too small, merge them
      repeat = False
      for a, b in zip(cut[:-1], cut[1:]):
        if (b - a) < min_sz:
          if a > 0:
            at = self.cad[a] - self.cad[a - 1]
          else:
            at = np.inf
          if b < len(self.cad) - 1:
            bt = self.cad[b + 1] - self.cad[b]
          else:
            bt = np.inf
          if at < bt:
            cut = np.delete(cut, np.where(cut == a))                                  # Merge with the closest chunk
          elif not np.isinf(bt):
            cut = np.delete(cut, np.where(cut == b))
          else:
            break
          repeat = True
          break
    self._jumps.extend(list(cut))
    self._jumps = list(set(self._jumps) - set([0, len(self.cad) - 1]))
    
    # Figure out the transit indices
    for ti in self.tN:
      i = np.where(np.abs(self.time - ti) <= self.tdur)[0]
      self._transits.extend(i)
    
    # Initialize our selectors
    self.Transits = RectangleSelector(ax, self.tselect,
                                      rectprops = dict(facecolor='green', 
                                                       edgecolor = 'black',
                                                       alpha=0.1, fill=True))
    self.Transits.set_active(False)
    self.Outliers = RectangleSelector(ax, self.oselect,
                                      rectprops = dict(facecolor='red', 
                                                       edgecolor = 'black',
                                                       alpha=0.1, fill=True))
    self.Outliers.set_active(False)  
    pl.connect('key_press_event', self.toggle)                                                 
    self.redraw(preserve_lims = False)
  
  def redraw(self, preserve_lims = True):
  
    # Reset the figure
    for p in self._plots:
      if len(p): p.pop(0).remove()
    self._plots = [p for p in self._plots if len(p)]
    self.ax.set_color_cycle(None)
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    ssi = [0] + sorted(self._jumps) + [len(self.cad) - 1]
    
    if self.state == 'fsum':
    
      # Replot stuff
      for a, b in zip(ssi[:-1], ssi[1:]):
        p = self.ax.plot(self.cad[a:b], self.fsum[a:b], '.')
        self._plots.append(p)
        color = p[0].get_color()
    
        # Transits
        x = []; y = []
        for ti in self.transits:
          if a <= ti and ti <= b:
            x.append(self.cad[ti])
            y.append(self.fsum[ti])
        p = self.ax.plot(x, y, '.', markersize = 15, color = color, zorder = -1, alpha = 0.25)
        self._plots.append(p)
    
      # Outliers
      o = self.outliers
      p = self.ax.plot(self.cad[o], self.fsum[o], 'ro')
      self._plots.append(p)
    
    elif self.state == 'fpix':
    
      for a, b in zip(ssi[:-1], ssi[1:]):
        for py in np.transpose(self.fpix):
          p = self.ax.plot(self.cad[a:b], py[a:b], '.')
          self._plots.append(p)
          color = p[0].get_color()
    
          # Transits
          x = []; y = []
          for ti in self.transits:
            if a <= ti and ti <= b:
              x.append(self.cad[ti])
              y.append(py[ti])
          p = self.ax.plot(x, y, '.', markersize = 15, color = color, zorder = -1, alpha = 0.25)
          self._plots.append(p)
    
          # Outliers
          o = self.outliers
          p = self.ax.plot(self.cad[o], py[o], 'ro')
          self._plots.append(p)
    
    # Splits
    for s in self._jumps:
      p = [self.ax.axvline((self.cad[s] + self.cad[s - 1]) / 2., color = 'k', 
                           ls = ':', alpha = 1.0)]
      self._plots.append(p)
    
    # Plot limits
    if preserve_lims:
      self.ax.set_xlim(xlim)
      self.ax.set_ylim(ylim)
    else:
    
      # X
      xmin = min(self.cad)
      xmax = max(self.cad)
      
      # Y
      if self.state == 'fpix':
        ymin = min([x for py in np.transpose(self.fpix) for x in py])
        ymax = max([x for py in np.transpose(self.fpix) for x in py])
      elif self.state == 'fsum':
        ymin = min(self.fsum)
        ymax = max(self.fsum)
      
      # Give it some margins 
      dx = (xmax - xmin) / 50.
      dy = (ymax - ymin) / 10.
      self.ax.set_xlim(xmin - dx, xmax + dx)    
      self.ax.set_ylim(ymin - dy, ymax + dy)
    
    # Labels
    self.ax.set_title(self.title, fontsize = 24)
    self.ax.set_xlabel('Cadence Number', fontsize = 22)
    self.ax.set_ylabel('Flux', fontsize = 22)
    
    # Refresh
    self.fig.canvas.draw()
  
  def get_inds(self, eclick, erelease):
    '''
    
    '''
        
    # Get coordinates
    x1 = min(eclick.xdata, erelease.xdata)
    x2 = max(eclick.xdata, erelease.xdata)
    y1 = min(eclick.ydata, erelease.ydata)
    y2 = max(eclick.ydata, erelease.ydata)
    
    if (x1 == x2) and (y1 == y2):   
      # User clicked 
      if self.state == 'fsum':
        d = ((self.cad - x1)/(max(self.cad) - min(self.cad))) ** 2 + ((self.fsum - y1)/(max(self.fsum) - min(self.fsum))) ** 2
        return [np.argmin(d)]
      elif self.state == 'fpix':
        da = [np.inf for i in range(self.fpix.shape[1])]
        ia = [0 for i in range(self.fpix.shape[1])]
        for i, p in enumerate(np.transpose(self.fpix)):
          d = ((self.cad - x1)/(max(self.cad) - min(self.cad))) ** 2 + ((p - y1)/(max(p) - min(p))) ** 2
          ia[i] = np.argmin(d)
          da[i] = d[ia[i]]
        return [min(zip(da, ia))[1]]
        
    else:
      # User selected
      if self.state == 'fsum':
        xy = zip(self.cad, self.fsum)
        return [i for i, pt in enumerate(xy) if x1<=pt[0]<=x2 and y1<=pt[1]<=y2] 
      elif self.state == 'fpix':
        inds = []
        for i, p in enumerate(np.transpose(self.fpix)):
          for i, xy in enumerate(zip(self.cad, p)):
            if (x1 <= xy[0] <= x2) and (y1 <= xy[1] <= y2):
              if i not in inds: 
                inds.append(i)
        return inds

  def tselect(self, eclick, erelease):
    '''
    
    '''
    
    inds = self.get_inds(eclick, erelease)
    for i in inds:
      if i in self._transits:
        self._transits.remove(i)
      else:
        self._transits.append(i)
    self.redraw()

  def oselect(self, eclick, erelease):
    '''
    
    '''
    
    inds = self.get_inds(eclick, erelease)
    for i in inds:
      if i in self._outliers:
        self._outliers.remove(i)
      else:
        self._outliers.append(i)   
    self.redraw()
    
  def toggle(self, event):
    
    # Transits
    if event.key == 't':
      self.Outliers.set_active(False)
      self.Transits.set_active(not self.Transits.active)
    
    # Outliers
    elif event.key == 'o':
      self.Transits.set_active(False)
      self.Outliers.set_active(not self.Outliers.active)
      
    # Splits
    elif event.key == 's':
      if event.inaxes is not None:
        s = np.argmax(self.cad == int(np.ceil(event.xdata)))
        
        # Are we in a gap?
        if self.cad[s] != int(np.ceil(event.xdata)):
          s1 = np.argmin(np.abs(self.cad - event.xdata))
          
          # Don't split at the end
          if s1 == len(self.cad) - 1:
            return
          
          s2 = s1 + 1
          if self.cad[s2] - self.cad[s1] > 1:
            s = s2
          else:
            s = s1

        # Don't split if s = 0
        if s == 0:
          return
        
        # Add
        if s not in self._jumps:
          self._jumps.append(s)
          
        # Delete
        else:
          i = np.argmin(np.abs(self.cad[self._jumps] - event.xdata))
          self._jumps.pop(i)
        
        self.redraw()
    
    # Toggle pixels
    if event.key == 'p':
      if self.state == 'fsum': self.state = 'fpix'
      elif self.state == 'fpix': self.state = 'fsum'
      self.redraw(preserve_lims = False)
      
    # Reset
    elif event.key == 'r':
     self.info = "reset"
      pl.close()

    # Next
    elif event.key == ' ' or event.key == 'enter' or event.key == 'right':
      self.info = "next"
      pl.close()
    
    # Previous
    elif event.key == 'left':
      self.info = "prev"
      pl.close()
    
    # Quit
    elif event.key == 'escape':
      self.info = "quit"
      pl.close()

  @property
  def outliers(self):
    return np.array(sorted(self._outliers), dtype = int)

  @property
  def transits(self):
    return np.array(sorted(self._transits), dtype = int)
  
  @property
  def jumps(self):
    return np.array(sorted(self._jumps), dtype = int)
  
def View(koi = 17.01, long_cadence = True, clobber = False,
         bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17], pad = 4.0,
         aperture = 'optimal', quarters = range(18), min_sz = 300,
         dt_tol = 0.5, split_cads = [4472, 6717], dir = config.dir, ttvs = False,
         quiet = False):
  '''
  
  '''

  # Grab the data
  if not quiet: print("Retrieving target data...")
  data, tN, per, tdur = GetTPFData(koi, long_cadence = long_cadence, clobber = clobber, dir = dir,
                        bad_bits = bad_bits, aperture = aperture, quarters = quarters,
                        quiet = quiet, pad = pad, ttvs = ttvs)
  data_new = EmptyData(quarters)
  data_trn = EmptyData(quarters)
  data_bkg = EmptyData(quarters)
  
  # Loop over all quarters
  if not quiet: print("Plotting...")
  uo = [[] for q in quarters]
  uj = [[] for q in quarters]
  ut = [[] for q in quarters]
  q = quarters[0]
  dq = 1
  while q in quarters:
  
    # Empty quarter?
    if data[q]['time'] == []:
      q += dq
      continue
      
    # Gap tolerance in cadences  
    cad_tol = dt_tol / np.median(data[q]['time'][1:] - data[q]['time'][:-1])
    
    # Set up the plot
    fig, ax = pl.subplots(1, 1, figsize = (16, 6))    
    sel = Selector(fig, ax, data[q]['cad'], data[q]['time'], data[q]['fsum'], 
                   data[q]['fpix'], tN, tdur, split_cads = split_cads, 
                   cad_tol = cad_tol, min_sz = min_sz, 
                   title = 'KOI %.2f: Quarter %02d' % (koi, q))
    
    # If user is re-visiting this quarter, update with their selections 
    if len(uj[q]): 
      sel._jumps = uj[q]
      sel.redraw()
    if len(uo[q]): 
      sel._outliers = uo[q]
      sel.redraw()
    if len(ut[q]): 
      sel._transits = ut[q]
      sel.redraw()
    
    pl.show()
    pl.close()
    
    # What will we do next time?
    if sel.info == "next":
      dq = 1
    elif sel.info == "prev":
      dq = -1
    elif sel.info == "reset":
      continue
    elif sel.info == "quit":
      return
    else:
      return
    
    # Store the user-defined outliers, jumps, and transits
    uo[q] = sel.outliers
    uj[q] = sel.jumps
    ut[q] = sel.transits
    

    # Get transits-only and background-only versions of the data
    inds = [i for i in range(len(data[q]['time'])) if i not in sel.outliers]
    tidx = [i for i in inds if i in sel.transits]
    bidx = [i for i in inds if i not in sel.transits]
    
    # If there's a jump across a transit, throw the transit out.
    bad = list(set(sel.jumps).intersection(tidx))
    
    # Median number of cadences per transit
    cpt = int(np.ceil(2 * tdur / np.median(data[q]['time'][1:] - data[q]['time'][:-1])))
    
    # Here are all the compromised transit indices
    for b in bad:
      for i in range(b - cpt, b + cpt):
        if i in tidx:
          tidx.remove(i)
    
    #
    # TODO: Split the data below!
    #
    
    
    # Remove outliers and split the data
    j = np.concatenate([[0], sel.jumps, [len(data[q]['time']) - 1]])
    o = sel.outliers
    for arr in ['time', 'fsum', 'ferr', 'fpix', 'perr', 'cad']:
      x = np.array(data[q][arr])
      x = np.delete(x, o, 0)
      for a, b in zip(j[:-1], j[1:]):
        data_new[q][arr].append(np.array(x[a:b]))
    
    
    # ???
    
    for arr in ['time', 'fsum', 'ferr', 'fpix', 'perr', 'cad']:
      data_trn[q][arr] = np.array(data[q][arr][tidx])
      data_bkg[q][arr] = np.array(data[q][arr][bidx])
      
    # Increment and loop
    q += dq
  
  # Save the data
  np.savez_compressed(os.path.join(dir, str(koi), 'data_proc.npz'), data = data_new)
  np.savez_compressed(os.path.join(dir, str(koi), 'data_trn.npz'), data = data_trn)
  np.savez_compressed(os.path.join(dir, str(koi), 'data_bkg.npz'), data = data_bkg)
    
  return

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-k", "--koi", default='254.01')
  args = parser.parse_args()
  View(koi = float(args.koi))