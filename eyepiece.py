#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
eyepiece.py
-----------

Download and visually inspect, split, and correct Kepler lightcurves.

'''

import matplotlib.pyplot as pl
import numpy as np
import kplr
import os

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

def GetTPFData(koi, long_cadence = True, clobber = False, 
               bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17],
               aperture = 'optimal', quarters = range(18), dir = '',
               quiet = False):
  '''
  
  '''
  
  if not clobber:
    try:
      data = np.load(os.path.join(dir, str(koi), 'data_raw.npz'))['data']
      if not quiet: print("Loading saved TPF data...")
      return data
    except:
      pass
  
  if not quiet: print("Downloading TPF data...")
  star = GetKoi(koi)
  data = EmptyData(quarters)
  tpf = star.get_target_pixel_files(short_cadence = not long_cadence)
  if len(tpf) == 0:
    raise Exception("No pixel files for selected object!")
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
  
  if not os.path.exists(os.path.join(dir, str(koi))):
    os.makedirs(os.path.join(dir, str(koi)))
  np.savez(os.path.join(dir, str(koi), 'data_raw.npz'), data = data)
  
  return data

def AddVLine(ax, t):
  '''
  
  '''
  return ax.axvline(t, color = 'k', ls = ':', alpha = 1.0)

class Interactor(object):
  '''
  
  '''
  def __init__(self, fig, ax, x, y, p, split_cads = [], cad_tol = 10, min_sz = 300,  
               style = '.', title = ''):
    self.fig = fig
    self.ax = ax
    self.x = np.array(x)
    self.y = np.array(y)
    self.p = np.array(p)
    self.style = style
    self.plots = [self.ax.plot(self.x, self.y, self.style)]
    self.odots = []
    self.fig.canvas.mpl_connect('key_press_event', self.on_key_press) 
    self.lines = []
    self.state = 'fsum'
    self._jumps = []
    self._outliers = np.array([], dtype = int)
    self.info = ""
    self.title = title
    
    # Add user-defined split locations
    for s in split_cads:
    
      # What if a split happens in a gap? Shift rightward up to 10 cadences
      for ds in range(10):
        foo = np.where(self.x == s + ds)[0]
        if len(foo):
          self._jumps.append(foo[0])
          break

    # Add cadence gap splits
    cut = np.where(self.x[1:] - self.x[:-1] > cad_tol)[0] + 1
    cut = np.concatenate([[0], cut, [len(self.x) - 1]])                               # Add first and last points
    repeat = True
    while repeat == True:                                                             # If chunks are too small, merge them
      repeat = False
      for a, b in zip(cut[:-1], cut[1:]):
        if (b - a) < min_sz:
          if a > 0:
            at = self.x[a] - self.x[a - 1]
          else:
            at = np.inf
          if b < len(self.x) - 1:
            bt = self.x[b + 1] - self.x[b]
          else:
            bt = np.inf
          if at < bt:
            cut = np.delete(cut, np.where(cut == a))                                  # Merge with the closest chunk
          elif not np.isinf(bt):
            cut = np.delete(cut, np.where(cut == b))
          repeat = True
          break
    self._jumps.extend(list(cut))
    self._jumps = np.array(list(set(self._jumps) - set([0, len(self.x) - 1])), dtype = int)
    
    self.redraw()
  
  def redraw(self):
  
    # Redraw the figure
    for plot in self.plots:
      plot.pop(0).remove()
    for odot in self.odots:
      odot.pop(0).remove()
    while len(self.lines): 
      self.lines[0].remove()
      del self.lines[0]
    pl.cla()
    self.ax.set_color_cycle(None)
      
    self.plots = []
    self.odots = []
    self.lines = []
    ssi = [0] + sorted(self._jumps) + [-1]

    # Plot the SAP flux
    if self.state == 'fsum':
    
      for a, b in zip(ssi[:-1], ssi[1:]):
        plot = self.ax.plot(self.x[a:b], self.y[a:b], self.style)
        self.plots.append(plot)
       
      for s in self._jumps:
        self.lines.append(AddVLine(self.ax, (self.x[s] + self.x[s - 1]) / 2.))

      for o in self._outliers:
        odot = self.ax.plot(self.x[o], self.y[o], 'ro')
        self.odots.append(odot)
    
    # Plot the pixel fluxes
    elif self.state == 'fpix':
    
      for a, b in zip(ssi[:-1], ssi[1:]):
        for p in np.transpose(self.p):
          plot = self.ax.plot(self.x[a:b], p[a:b], self.style)
          self.plots.append(plot)
       
      for s in self._jumps:
        self.lines.append(AddVLine(self.ax, (self.x[s] + self.x[s - 1]) / 2.))

      for o in self._outliers:
        for p in np.transpose(self.p):
          odot = self.ax.plot(self.x[o], p[o], 'ro')
        self.odots.append(odot)
        
    self.ax.margins(0.01,0.1)
    
    self.ax.set_title(self.title, fontsize = 24)
    self.ax.set_xlabel('Cadence Number', fontsize = 22)
    self.ax.set_ylabel('Flux', fontsize = 22)
    
    self.fig.canvas.draw()
    
  def on_key_press(self, event):
  
    # Split the lightcurve
    if event.key == 'x':
      if event.inaxes is not None:
        s = np.argmax(self.x == int(np.ceil(event.xdata)))
        
        # Are we in a gap?
        if self.x[s] != int(np.ceil(event.xdata)):
          s1 = np.argmin(np.abs(self.x - event.xdata))
          
          # Don't split at the end
          if s1 == len(self.x) - 1:
            return
          
          s2 = s1 + 1
          if self.x[s2] - self.x[s1] > 1:
            s = s2
          else:
            s = s1

        # Don't split if s = 0
        if s == 0:
          return
        
        # Add
        if s not in self._jumps:
          self._jumps = np.append(self._jumps, s)
          
        # Delete
        else:
          i = np.argmin(np.abs(self.x[self._jumps] - event.xdata))
          self._jumps = np.delete(self._jumps, i)
        
        self.redraw()
    
    # Toggle outliers
    if event.key == 'o':
      if event.inaxes is not None:
        x = event.xdata
        y = event.ydata
        
        if self.state == 'fsum':
          s = np.argmin((x - self.x) ** 2 + (y - self.y) ** 2)
        elif self.state == 'fpix':
          s = np.argmin((x - self.x) ** 2)
        
        # Add
        if s not in self._outliers:
          self._outliers = np.append(self._outliers, s)
        
        # Delete
        else:
          self._outliers = np.delete(self._outliers, np.argmax(self._outliers == s))
        
        self.redraw()
    
    # Toggle pixels
    if event.key == 'p':
      if self.state == 'fsum': self.state = 'fpix'
      elif self.state == 'fpix': self.state = 'fsum'
      self.redraw()
    
    # Reset
    if event.key == 'r':
     self._jumps = np.array([], dtype = int)
     self._outliers = np.array([], dtype = int)
     self.redraw()
        
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
  def jumps(self):
    return np.array(sorted(self._jumps), dtype = int)
  
def View(koi = 17.01, long_cadence = True, clobber = False,
         bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17],
         aperture = 'optimal', quarters = range(18), min_sz = 300,
         dt_tol = 0.5, split_cads = [4472, 6717], dir = '',
         quiet = False):
  '''
  
  '''

  # Grab the data
  if not quiet: print("Retrieving target data...")
  data = GetTPFData(koi, long_cadence = long_cadence, clobber = clobber, dir = dir,
                    bad_bits = bad_bits, aperture = aperture, quarters = quarters,
                    quiet = quiet)
  data_new = EmptyData(quarters)
  
  # Loop over all quarters
  if not quiet: print("Plotting...")
  uo = [[] for q in quarters]
  uj = [[] for q in quarters]
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
    itr = Interactor(fig, ax, data[q]['cad'], data[q]['fsum'], data[q]['fpix'], 
                     min_sz = min_sz, split_cads = split_cads, cad_tol = cad_tol,
                     title = 'KOI %.2f: Quarter %02d' % (koi, q))
    
    # If user is re-visiting this quarter, update with their selections 
    if len(uj[q]): 
      itr._jumps = uj[q]
      itr.redraw()
    if len(uo[q]): 
      itr._outliers = uo[q]
      itr.redraw()
    
    pl.show()
    pl.close()
    
    # Store the user-defined outliers and jumps
    uo[q] = itr.outliers
    uj[q] = itr.jumps
    
    # Remove outliers and split the data
    j = np.concatenate([[0], itr.jumps, [len(data[q]['time']) - 1]])
    o = itr.outliers
    for arr in ['time', 'fsum', 'ferr', 'fpix', 'perr', 'cad']:
      x = np.array(data[q][arr])
      x = np.delete(x, o, 0)
      for a, b in zip(j[:-1], j[1:]):
        data_new[q][arr].append(np.array(x[a:b]))

    # What shall we do next?
    if itr.info == "next":
      dq = 1
    elif itr.info == "prev":
      dq = -1
    elif itr.info == "quit":
      return
  
    q += dq
  
  # Save the data
  np.savez(os.path.join(dir, str(koi), 'data_split.npz'), data = data_new)
    
  return

if __name__ == '__main__':
  View()