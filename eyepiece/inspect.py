#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
inspect.py
----------

Download and visually inspect, split, and correct Kepler lightcurves.

.. todo::
   - Use ``blit`` for faster redrawing!
   - Allow user to select ``tdur`` and ``pad``
   - Suppress this message: ``setCanCycle: is deprecated.  Please use setCollectionBehavior instead``
   - Transit utility with outlier selection!
   - Bring focus to plot
   - Errorbars
   - Show transit numbers
   - Show quarters in transit plot

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from .utils import Bold, RowCol, GitHash
from .download import GetTPFData, GetData, EmptyData
import eyepiece.config as config
import matplotlib.pyplot as pl
from matplotlib.widgets import RectangleSelector, Cursor
from matplotlib import patches
import numpy as np
import os
import itertools

__all__ = ["Inspect", "PlotTransits"]

def GetJumps(time, cad, cad_tol = 10, min_sz = 300, split_cads = []):
  '''
  
  '''
  
  # Initialize
  jumps = []
  transits = []
  
  # Add user-defined split locations
  for s in split_cads:
  
    # What if a split happens in a gap? Shift rightward up to 10 cadences
    for ds in range(10):
      foo = np.where(cad == s + ds)[0]
      if len(foo):
        jumps.append(foo[0])
        break
  
  # Add cadence gap splits
  cut = np.where(cad[1:] - cad[:-1] > cad_tol)[0] + 1
  cut = np.concatenate([[0], cut, [len(cad) - 1]])                                    # Add first and last points
  repeat = True
  while repeat == True:                                                               # If chunks are too small, merge them
    repeat = False
    for a, b in zip(cut[:-1], cut[1:]):
      if (b - a) < min_sz:
        if a > 0:
          at = cad[a] - cad[a - 1]
        else:
          at = np.inf
        if b < len(cad) - 1:
          bt = cad[b + 1] - cad[b]
        else:
          bt = np.inf
        if at < bt:
          cut = np.delete(cut, np.where(cut == a))                                    # Merge with the closest chunk
        elif not np.isinf(bt):
          cut = np.delete(cut, np.where(cut == b))
        else:
          break
        repeat = True
        break
  jumps.extend(list(cut))
  jumps = list(set(jumps) - set([0, len(cad) - 1]))
  
  return jumps

class DummySelector(object):
  '''
  
  '''
  
  def __init__(self):
    self.active = True
  
  def set_active(self, active):
    self.active = active

class Selector(object):
  '''
  
  '''
  
  def __init__(self, fig, ax, koi, quarter, data, tN, tdur, pad = 2.0, split_cads = [], 
               cad_tol = 10, min_sz = 300, dir = config.dir):
    self.fig = fig
    self.ax = ax
    self.cad = data['cad']
    self.time = data['time']
    self.fsum = data['fsum']
    self.fpix = data['fpix']
    self.tNc = self.cad[0] + (self.cad[-1] - self.cad[0])/(self.time[-1] - self.time[0]) * (tN - self.time[0])
    self.koi = koi
    self.dir = dir
    self.quarter = quarter
    self.title = 'KOI %.2f: Quarter %02d' % (self.koi, self.quarter)
    
    # Initialize arrays
    self._outliers = []
    self._transits = []
    self._jumps = []
    self._plots = []
    self._transit_lines = []
    self.info = ""
    self.state = "fsum"
    self.alt = False
    self.hide = False
    
    # Process the data
    self._jumps = GetJumps(self.time, self.cad, cad_tol = cad_tol, min_sz = min_sz, 
                           split_cads = split_cads)
    
    # Time per cadence
    tpc = np.median(self.time[1:] - self.time[:-1])
    # Cadences per transit
    self.cpt = (pad * tdur / tpc)
    
    # Transit indices
    self._transits = []
    for tc in self.tNc:
      i = np.where(np.abs(self.cad - tc) <= self.cpt / 2.)[0]
      self._transits.extend(i) 
  
    # Initialize our selectors
    self.Outliers = RectangleSelector(ax, self.oselect,
                                      rectprops = dict(facecolor='red', 
                                                       edgecolor = 'black',
                                                       alpha=0.1, fill=True))
    self.Outliers.set_active(False)
    self.Zoom = RectangleSelector(ax, self.zselect,
                                  rectprops = dict(edgecolor = 'black', linestyle = 'dashed',
                                                   fill=False))
    self.Zoom.set_active(False) 
    self.Transits = DummySelector()
    self.Transits.set_active(False)
    self.Split = DummySelector()
    self.Split.set_active(False)
    
    pl.connect('key_press_event', self.on_key_press)
    pl.connect('key_release_event', self.on_key_release) 
    pl.connect('motion_notify_event', self.on_mouse_move)     
    pl.connect('button_release_event', self.on_mouse_release)  
    pl.connect('button_press_event', self.on_mouse_click)                                         
    self.redraw(preserve_lims = False)
  
  def ShowHelp(self):
    '''
    
    '''
    
    print("")
    print(Bold("Eyepiece v0.0.1 Help"))
    print("")
    commands = {'h': 'Display this ' + Bold('h') + 'elp message',
                'z': 'Toggle ' + Bold('z') + 'ooming',
                'b': Bold('B') + 'ack to original view',
                'p': 'Toggle ' + Bold('p') + 'ixel view',
                't': 'Toggle ' + Bold('t') + 'ransit selection tool',
                'o': 'Toggle ' + Bold('o') + 'utlier selection tool',
                'x': 'Toggle transits and outliers on/off',
                's': 'Toggle lightcurve ' + Bold('s') + 'plitting tool',
                '←': 'Go to previous quarter',
                '→': 'Go to next quarter',
                'r': Bold('R') + 'eset all selections',
                'Esc': 'Quit (without saving)',
                'Ctrl+s': Bold('S') + 'ave current plot to disk'
                }
    for key, descr in sorted(commands.items()):
      print('%s:\t%s' % (Bold(key), descr))
    print("")
  
  def redraw(self, preserve_lims = True):
  
    # Reset the figure
    for p in self._plots:
      if len(p): p.pop(0).remove()
    self._plots = []
    for l in self._transit_lines:
      for i in l:
        if len(i): i.pop(0).remove()
    self._transit_lines = []
    self.ax.set_color_cycle(None)
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    ssi = [0] + sorted(self._jumps) + [len(self.cad) - 1]
    
    if self.state == 'fsum':
    
      # Replot stuff
      for a, b in zip(ssi[:-1], ssi[1:]):
        
        # Non-transit, non-outlier inds
        inds = [i for i, c in enumerate(self.cad) if (c >= self.cad[a]) and (c < self.cad[b]) and (i not in self.transits) and (i not in self.outliers)]
        p = self.ax.plot(self.cad[inds], self.fsum[inds], '.')
        self._plots.append(p)
        color = p[0].get_color()
    
        if not self.hide:
          # Transits
          x = []; y = []
          for ti in self.transits:
            if a <= ti and ti <= b:
              x.append(self.cad[ti])
              y.append(self.fsum[ti])
          p = self.ax.plot(x, y, '.', color = color)    
          self._plots.append(p)
          p = self.ax.plot(x, y, '.', markersize = 15, color = color, zorder = -1, alpha = 0.25)
          self._plots.append(p)
    
      if not self.hide:
        # Outliers
        o = self.outliers
        p = self.ax.plot(self.cad[o], self.fsum[o], 'ro')
        self._plots.append(p)
    
    elif self.state == 'fpix':
    
      for a, b in zip(ssi[:-1], ssi[1:]):
        for py in np.transpose(self.fpix):
          # Non-transit, non-outlier inds
          inds = [i for i, c in enumerate(self.cad) if (c >= self.cad[a]) and (c < self.cad[b]) and (i not in self.transits) and (i not in self.outliers)]
          p = self.ax.plot(self.cad[inds], py[inds], '.')
          self._plots.append(p)
          color = p[0].get_color()
    
          if not self.hide:
            # Transits
            x = []; y = []
            for ti in self.transits:
              if a <= ti and ti <= b:
                x.append(self.cad[ti])
                y.append(py[ti])
            p = self.ax.plot(x, y, '.', color = color)
            self._plots.append(p)
            p = self.ax.plot(x, y, '.', markersize = 15, color = color, zorder = -1, alpha = 0.25)
            self._plots.append(p)
    
          if not self.hide:
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
    
    # Transit times
    if not self.hide:
      for tc in self.tNc:
        if self.cad[0] <= tc <= self.cad[-1]:
          l1 = [self.ax.axvline(tc, color = 'r', alpha = 0.1, ls = '-')]
          l2 = self.ax.plot(tc, self.ax.get_ylim()[1], 'rv', markersize = 15)
          l3 = self.ax.plot(tc, self.ax.get_ylim()[0], 'r^', markersize = 15)
          self._transit_lines.append([l1, l2, l3])
    
    # Labels
    self.ax.set_title(self.title, fontsize = 24, y = 1.01)
    self.ax.set_xlabel('Cadence Number', fontsize = 22)
    self.ax.set_ylabel('Flux', fontsize = 22)
    
    # Operations
    label = None
    if self.Zoom.active:
      label = 'zoom'
    elif self.Outliers.active:
      if self.alt:
        label = 'outliers (-)'
      else:
        label = 'outliers (+)'
    elif self.Transits.active:
      label = 'transits'
    elif self.Split.active:
      label = 'split'
    
    if label is not None:
      a = self.ax.text(0.005, 0.96, label, fontsize=12, transform=self.ax.transAxes, color = 'r', alpha = 0.75)
      self._plots.append([a])
      
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

  def zselect(self, eclick, erelease):
    '''
    
    '''
    
    if not (eclick.xdata == erelease.xdata and eclick.ydata == erelease.ydata):
      self.ax.set_xlim(min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata))
      self.ax.set_ylim(min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata))
      self.redraw()

  def oselect(self, eclick, erelease):
    '''
    
    '''
    
    inds = self.get_inds(eclick, erelease)
    for i in inds:
      if self.alt:
        if i in self._outliers:
          self._outliers.remove(i)
      else:
        if i not in self._outliers:
          self._outliers.append(i)   
    self.redraw()
  
  def on_key_release(self, event):
    
    # Alt
    if event.key == 'alt':
      self.alt = False
      self.redraw()
      
  def on_key_press(self, event):
  
    # Super (command)
    if event.key == 'super+s':
      figname = os.path.join(self.dir, str(self.koi), "Q%02d_%s.png" % (self.quarter, self.state))
      self.fig.savefig(figname, bbox_inches = 'tight')
      print("Saved to file %s." % figname)
  
    # Alt
    if event.key == 'alt':
      self.alt = True
      self.redraw()
    
    # Home (back)
    if event.key == 'b':
      self.redraw(preserve_lims = False)
    
    # Help
    if event.key == 'h':
      self.ShowHelp()
    
    # Hide/Show
    if event.key == 'x':
      self.hide = not self.hide
      self.redraw()
    
    # Zoom
    if event.key == 'z':
      self.Transits.set_active(False)
      self.Outliers.set_active(False)
      self.Split.set_active(False)
      self.Zoom.set_active(not self.Zoom.active)
      self.redraw()
    
    # Transits
    if event.key == 't':
      self.Outliers.set_active(False)
      self.Zoom.set_active(False)
      self.Split.set_active(False)
      self.Transits.set_active(not self.Transits.active)
      self.redraw()

    # Outliers
    elif event.key == 'o':
      self.Transits.set_active(False)
      self.Zoom.set_active(False)
      self.Split.set_active(False)
      self.Outliers.set_active(not self.Outliers.active)
      self.redraw()
      
    # Splits
    elif event.key == 's':
      self.Transits.set_active(False)
      self.Zoom.set_active(False)
      self.Outliers.set_active(False)
      self.Split.set_active(not self.Outliers.active)
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
  
  def on_mouse_release(self, event):
    if self.Transits.active:
      self.redraw()
  
  def on_mouse_click(self, event):
    if (event.inaxes is not None) and (self.Split.active):
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
  
  def on_mouse_move(self, event):
    if (self.hide) or (not self.Transits.active) or (event.button != 1): 
      return
    x, y = event.xdata, event.ydata
    i = np.argmin(np.abs(x - self.cad))
    j = np.argmin(np.abs(self.cad[i] - self.tNc))
    self.tNc[j] = x
    
    # Remove these from the plot
    for l in self._transit_lines:
      for i in l:
        if len(i): i.pop(0).remove()
    self._transit_lines = []
    
    # Figure out the transit indices
    self._transits = []
    for tc in self.tNc:
      i = np.where(np.abs(self.cad - tc) <= self.cpt / 2.)[0]
      self._transits.extend(i)
    
      if self.cad[0] <= tc <= self.cad[-1]:
        l1 = [self.ax.axvline(tc, color = 'r', alpha = 0.1, ls = '-')]
        l2 = self.ax.plot(tc, self.ax.get_ylim()[1], 'rv', markersize = 15)
        l3 = self.ax.plot(tc, self.ax.get_ylim()[0], 'r^', markersize = 15)
        self._transit_lines.append([l1, l2, l3])
    
    self.fig.canvas.draw()
  
  @property
  def tN(self):
    return self.time[0] + (self.time[-1] - self.time[0])/(self.cad[-1] - self.cad[0]) * (self.tNc - self.cad[0])
  
  @property
  def outliers(self):
    return np.array(sorted(self._outliers), dtype = int)

  @property
  def transits(self):
    return np.array(sorted(self._transits), dtype = int)
  
  @property
  def jumps(self):
    return np.array(sorted(self._jumps), dtype = int)
  
def Inspect(koi = 17.01, long_cadence = True, clobber = False,
            bad_bits = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17], padbkg = 2.0,
            padtrn = 5.0, aperture = 'optimal', quarters = range(18), min_sz = 300,
            dt_tol = 0.5, split_cads = [4472, 6717], dir = config.dir, ttvs = False,
            quiet = False, blind = False):
      '''
  
      '''

      # Grab the data
      if not quiet: print("Retrieving target data...")
      data, tN, per, tdur = GetTPFData(koi, long_cadence = long_cadence, 
                            clobber = clobber, dir = dir, bad_bits = bad_bits, 
                            aperture = aperture, quarters = quarters,
                            quiet = quiet, pad = padbkg, ttvs = ttvs)
      data_new = EmptyData(quarters)
      data_trn = EmptyData(quarters)
      data_bkg = EmptyData(quarters)
  
      # Loop over all quarters
      if not quiet: print("Inspecting...")
      uo = [[] for q in quarters]
      uj = [[] for q in quarters]
      ut = [[] for q in quarters]
      q = quarters[0]
      dq = 1
      while q in quarters:
        
        if not quiet: print("Quarter %d" % q)
  
        # Empty quarter?
        if data[q]['time'] == []:
          q += dq
          continue
      
        # Gap tolerance in cadences  
        cad_tol = dt_tol / np.median(data[q]['time'][1:] - data[q]['time'][:-1])
    
        if not blind:
        
          # Set up the plot
          fig, ax = pl.subplots(1, 1, figsize = (16, 6))
          fig.subplots_adjust(top=0.9, bottom=0.15, left = 0.075, right = 0.95)   
          sel = Selector(fig, ax, koi, q, data[q], tN, tdur, pad = padbkg, 
                       split_cads = split_cads, cad_tol = cad_tol, min_sz = min_sz)
                    
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
    
          fig.canvas.set_window_title('Eyepiece')       
          
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
          
          jumps = sel.jumps
          transits = sel.transits
          outliers = sel.outliers
          tN = sel.tN
        
        else:
          
          # Just process the data
          jumps, transits = Process(data[q]['time'], data[q]['cad'], tN, tdur, 
                                    pad = padbkg, cad_tol = cad_tol, min_sz = min_sz, 
                                    split_cads = split_cads)
          
          # Process the data
          jumps = GetJumps(data[q]['time'], data[q]['cad'], cad_tol = cad_tol, 
                           min_sz = min_sz, split_cads = split_cads)
          # Time per cadence
          tpc = np.median(data[q]['time'][1:] - data[q]['time'][:-1])
          # Cadences per transit
          cpt = (padbkg * tdur / tpc)
         
          # Transit indices. TODO: Verify
          transits = []
          tNc = data[q]['cad'][0] + (data[q]['cad'][-1] - data[q]['cad'][0])/(data[q]['time'][-1] - data[q]['time'][0]) * (tN - data[q]['time'][0])
          for tc in tNc:
            i = np.where(np.abs(data[q]['cad'] - tc) <= cpt / 2.)[0]
            transits.extend(i) 
          
          jumps = np.array(jumps, dtype = int)
          transits = np.array(transits, dtype = int)
          outliers = np.array([], dtype = int)
        
        # Store the user-defined outliers, jumps, and transits
        uo[q] = outliers
        uj[q] = jumps
        ut[q] = transits
  
        # Split the data
        j = np.concatenate([[0], jumps, [len(data[q]['time']) - 1]])
        for arr in ['time', 'fsum', 'ferr', 'fpix', 'perr', 'cad']:
      
          # All data and background data
          for a, b in zip(j[:-1], j[1:]):
      
            ai = [i for i in range(a, b) if i not in outliers]
            bi = [i for i in range(a, b) if i not in np.append(outliers, transits)]
      
            data_new[q][arr].append(np.array(data[q][arr][ai]))
            data_bkg[q][arr].append(np.array(data[q][arr][bi]))
      
          # Transit data
          for t in tN:
            ti = list( np.where( np.abs(data[q]['time'] - t) < (padtrn * tdur) / 2. )[0] )
            ti = [i for i in ti if i not in outliers]
        
            # If there's a jump across this transit, throw it out.
            if len(list(set(jumps).intersection(ti))) == 0 and len(ti) > 0:
              data_trn[q][arr].append(data[q][arr][ti])
          
        # Increment and loop
        q += dq
  
      # Save the data
      if not quiet: print("Saving data to disk...")
      np.savez_compressed(os.path.join(dir, str(koi), 'data_proc.npz'), data = data_new, hash = GitHash())
      np.savez_compressed(os.path.join(dir, str(koi), 'data_trn.npz'), data = data_trn, hash = GitHash())
      np.savez_compressed(os.path.join(dir, str(koi), 'data_bkg.npz'), data = data_bkg, hash = GitHash())

      return

def PlotTransits(koi = 17.01, quarters = range(18), dir = config.dir, ttvs = False):
  '''
  
  '''
  time = []
  flux = []
  
  try:
    data = np.load(os.path.join(dir, str(koi), 'data_trn.npz'))['data']
    foo = np.load(os.path.join(dir, str(koi), 'transits.npz'))
    tN = foo['tN']
    per = foo['per']
    tdur = foo['tdur']
  except IOError:
    raise Exception("You must download and process the data first! Try using ``Inspect()``.")
  
  for q in quarters:
    for t, f in zip(data[q]['time'], data[q]['fsum']):
      time.append(t)
      flux.append(f)
    
  COLS, ROWS = RowCol(len(time))
  grid = list(itertools.product(*[np.arange(COLS), np.arange(ROWS)]))  
  fig, axes = pl.subplots(COLS, ROWS, figsize = (2.5*ROWS,2.5*COLS))
  fig.subplots_adjust(wspace=0.05, hspace=0.05)
  if (COLS*ROWS) > len(time):
    for g in grid[len(time):]:
      axes[g].set_visible(False)
  for i, _ in enumerate(time):
  
    # What transit number is this?
    tnum = np.argmin(np.abs(tN - time[i][0]))
    tNi = tN[tnum]

    if (COLS > 1):
      ax = axes[grid[i]]
    else:
      ax = axes[grid[i][1]]

    ax.plot(time[i], flux[i], 'b.')
    ax.plot(time[i], flux[i], 'b-', alpha = 0.25)
    ax.annotate("%03d" % tnum,
            xy=(0.8, 0.05), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points')
    ax.axvline(tNi, color = 'r', ls = '-', alpha = 0.75)
    ax.set_xlim(tNi - tdur / 2., tNi + tdur / 2.)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
  
  fig.savefig(os.path.join(dir, str(koi), "transits.png"), bbox_inches = 'tight')
  pl.close()