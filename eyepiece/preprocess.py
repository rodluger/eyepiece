#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
preprocess.py
-------------

.. todo::
   - Add keyboard shortcut to view PDC data

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .utils import Bold, GitHash, Input, GetData
from .download import DownloadData, DownloadInfo, EmptyData
import numpy as np
import os

# Python 2/3 compatibility.
import sys
if sys.version_info >= (3,0):
	prompt = input
else:
  prompt = raw_input
  
rcParams = None

# Keyboard shortcuts
ZOOM = 'z'
QUIT = 'escape'
HOME = 'H'
HELP = 'h'
PIXL = 'p'
RSET = 'r'
SPLT = 's'
TRAN = 't'
OUTL = 'o'
HIDE = 'x'
PREV = 'left'
NEXT = 'right'
SUBT = 'alt'
TPAD = 'w'
ERRB = 'e'
BLND = 'b'

def DisableShortcuts():
  '''
  Disable MPL keyboard shortcuts and the plot toolbar.

  '''
  
  global rcParams
  rcParams = dict(pl.rcParams)
    
  pl.rcParams['toolbar'] = 'None'
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

def EnableShortcuts():
  '''
  Resets pl.rcParams to its original state.
  
  '''
  
  global rcParams
  
  pl.rcParams.update(rcParams)

def ShowHelp():
  '''
  
  '''
  
  print("")
  print(Bold("Eyepiece v0.0.1 Help"))
  print("")
  commands = {HELP: 'Display this help message',
              ZOOM: 'Toggle zooming',
              HOME: 'Home view',
              PIXL: 'Toggle pixel view',
              TRAN: 'Toggle transit selection tool',
              OUTL: 'Toggle outlier selection tool',
              HIDE: 'Toggle transits and outliers on/off',
              SPLT: 'Toggle lightcurve splitting tool',
              PREV: 'Go to previous quarter',
              NEXT: 'Go to next quarter',
              RSET: 'Reset all selections',
              QUIT: 'Quit (without saving)',
              TPAD: 'Adjust transit padding',
              ERRB: 'Show/hide error bars',
              BLND: 'Continue in blind mode'
              }
  for key, descr in sorted(commands.items()):
    print('%s%s' % (Bold(key + ':').ljust(30), descr))
  print("")
  prompt("Press Enter to continue...")

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

class Viewer(object):
  '''
  
  '''
  
  def __init__(self, fig, ax, id, quarter, data, tN, cptbkg, cpttrn, datadir, 
               split_cads = [], cad_tol = 10, min_sz = 300, interactive = True):
    self.fig = fig
    self.ax = ax    
    self.cad = data['cadn']
    self.time = data['time']
    self.fsum = np.sum(data['fpix'], axis = 1)
    self.ferr = np.sqrt(np.sum(data['perr'] ** 2, axis = 1))
    self.perr = data['perr']
    self.fpix = data['fpix']
    self.tNc = self.cad[0] + (self.cad[-1] - self.cad[0])/(self.time[-1] - self.time[0]) * (tN - self.time[0])
    self.id = id
    self.datadir = datadir
    self.quarter = quarter
    self.interactive = interactive
    
    # Initialize arrays
    self._outliers = []
    self._transits_narrow = []
    self._transits_wide = []
    self._jumps = []
    self._plots = []
    self._transit_lines = []
    self.info = ""
    self.state = ["fsum", "none"]
    self.subt = False
    self.hide = False
    
    # Process the data
    self._jumps = GetJumps(self.time, self.cad, cad_tol = cad_tol, min_sz = min_sz, 
                           split_cads = split_cads)

    # Cadences per transit
    self.cptbkg = cptbkg
    self.cpttrn = cpttrn
    
    # Transit indices
    self._transits_narrow = []
    self._transits_wide = []
    self._transits_wide_tag = []
    for j, tc in enumerate(self.tNc):
      i = np.where(np.abs(self.cad - tc) <= self.cptbkg / 2.)[0]
      self._transits_narrow.extend(i) 
      i = np.where(np.abs(self.cad - tc) <= self.cpttrn / 2.)[0]
      self._transits_wide.extend(i) 
      self._transits_wide_tag.extend([j for k in i])
    
    # Interactive features
    if self.interactive:
      
      from matplotlib.widgets import RectangleSelector
      
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
    
    else:
      
      # Dummy selectors (serve no purpose)
      self.Outliers = DummySelector()
      self.Outliers.set_active(False)
      self.Zoom = DummySelector()
      self.Zoom.set_active(False)
      self.Transits = DummySelector()
      self.Transits.set_active(False)
      self.Split = DummySelector()
      self.Split.set_active(False)
      
      self.info = "next"
                                         
    self.redraw(preserve_lims = False)
  
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
    
    if self.state[0] == 'fsum':
    
      # Replot stuff
      for a, b in zip(ssi[:-1], ssi[1:]):
        
        # Non-transit, non-outlier inds
        inds = [i for i, c in enumerate(self.cad) if (c >= self.cad[a]) and (c < self.cad[b]) and (i not in self.transits_narrow) and (i not in self.outliers)]
        p = self.ax.plot(self.cad[inds], self.fsum[inds], '.')
        self._plots.append(p)
        color = p[0].get_color()
    
        if not self.hide:
          # Transits_narrow
          x = []; y = []
          for ti in self.transits_narrow:
            if a <= ti and ti <= b:
              x.append(self.cad[ti])
              y.append(self.fsum[ti])
          p = self.ax.plot(x, y, '.', color = color)    
          self._plots.append(p)
          p = self.ax.plot(x, y, 'o', markersize = 10, color = color, zorder = -1, alpha = 0.25)
          self._plots.append(p)
          
          # Transits_wide
          x = []; y = []
          for ti in [foo for foo in self.transits_wide if foo not in self.transits_narrow]:
            if a <= ti and ti <= b:
              x.append(self.cad[ti])
              y.append(self.fsum[ti])
          p = self.ax.plot(x, y, 'o', markersize = 10, markerfacecolor = 'None', markeredgecolor = color, zorder = -1, alpha = 0.1)
          self._plots.append(p)
    
      if not self.hide:
        # Outliers
        o = self.outliers
        p = self.ax.plot(self.cad[o], self.fsum[o], 'ro')
        self._plots.append(p)
    
      if self.state[1] == 'errorbars':
        p = self.ax.errorbar(self.cad, self.fsum, self.ferr, fmt='none', ecolor='k', capsize=0)
        self._plots.append([p[2][0]])
    
    elif self.state[0] == 'fpix':
    
      for a, b in zip(ssi[:-1], ssi[1:]):
        for py, pyerr in zip(np.transpose(self.fpix), np.transpose(self.perr)):
          # Non-transit, non-outlier inds
          inds = [i for i, c in enumerate(self.cad) if (c >= self.cad[a]) and (c < self.cad[b]) and (i not in self.transits_narrow) and (i not in self.outliers)]
          p = self.ax.plot(self.cad[inds], py[inds], '.')
          self._plots.append(p)
          color = p[0].get_color()
    
          if not self.hide:
            # Transits
            x = []; y = []
            for ti in self.transits_narrow:
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
    
          if self.state[1] == 'errorbars':
            p = self.ax.errorbar(self.cad, py, pyerr, fmt='none', ecolor='k', capsize=0)
            self._plots.append([p[2][0]])  
          
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
      if self.state[0] == 'fpix':
        ymin = min([x for py in np.transpose(self.fpix) for x in py])
        ymax = max([x for py in np.transpose(self.fpix) for x in py])
      elif self.state[0] == 'fsum':
        ymin = min(self.fsum)
        ymax = max(self.fsum)
      
      # Give it some margins 
      dx = (xmax - xmin) / 50.
      dy = (ymax - ymin) / 10.
      self.ax.set_xlim(xmin - dx, xmax + dx)    
      self.ax.set_ylim(ymin - dy, ymax + dy)
    
    # Transit times
    if not self.hide:
      for i, tc in enumerate(self.tNc):
        if self.cad[0] <= tc <= self.cad[-1]:
          l1 = [self.ax.axvline(tc, color = 'r', alpha = 0.1, ls = '-')]
          l2 = self.ax.plot(tc, self.ax.get_ylim()[1], 'rv', markersize = 15)
          l3 = self.ax.plot(tc, self.ax.get_ylim()[0], 'r^', markersize = 15)
          self._transit_lines.append([l1, l2, l3])
          yl = self.ax.get_ylim()
          y = yl[1] + 0.01 * (yl[1] - yl[0])
          a = self.ax.text(tc, y, i, ha = 'center', fontsize = 8, alpha = 0.5)
          self._plots.append([a])
    
    # Labels
    self.ax.set_xlabel('Cadence Number', fontsize = 22)
    self.ax.set_ylabel('Flux', fontsize = 22)
    
    # Operations
    label = None
    if self.Zoom.active:
      label = 'zoom'
    elif self.Outliers.active:
      if self.subt:
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
      if self.state[0] == 'fsum':
        d = ((self.cad - x1)/(max(self.cad) - min(self.cad))) ** 2 + ((self.fsum - y1)/(max(self.fsum) - min(self.fsum))) ** 2
        return [np.argmin(d)]
      elif self.state[0] == 'fpix':
        da = [np.inf for i in range(self.fpix.shape[1])]
        ia = [0 for i in range(self.fpix.shape[1])]
        for i, p in enumerate(np.transpose(self.fpix)):
          d = ((self.cad - x1)/(max(self.cad) - min(self.cad))) ** 2 + ((p - y1)/(max(p) - min(p))) ** 2
          ia[i] = np.argmin(d)
          da[i] = d[ia[i]]
        return [min(zip(da, ia))[1]]
        
    else:
      # User selected
      if self.state[0] == 'fsum':
        xy = zip(self.cad, self.fsum)
        return [i for i, pt in enumerate(xy) if x1<=pt[0]<=x2 and y1<=pt[1]<=y2] 
      elif self.state[0] == 'fpix':
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
      if self.subt:
        if i in self._outliers:
          self._outliers.remove(i)
      else:
        if i not in self._outliers:
          self._outliers.append(i)   
    self.redraw()
  
  def on_key_release(self, event):
    
    # Alt
    if event.key == SUBT:
      self.subt = False
      self.redraw()
      
  def on_key_press(self, event):
    
    # Alt
    if event.key == SUBT:
      self.subt = True
      self.redraw()
    
    # Home
    if event.key == HOME:
      self.redraw(preserve_lims = False)
    
    # Help
    if event.key == HELP:
      self.info = "help"
      pl.close()
    
    # Hide/Show
    if event.key == HIDE:
      self.hide = not self.hide
      self.redraw()
    
    # Zoom
    if event.key == ZOOM:
      self.Transits.set_active(False)
      self.Outliers.set_active(False)
      self.Split.set_active(False)
      self.Zoom.set_active(not self.Zoom.active)
      self.redraw()
    
    # Transits
    if event.key == TRAN:
      self.Outliers.set_active(False)
      self.Zoom.set_active(False)
      self.Split.set_active(False)
      self.Transits.set_active(not self.Transits.active)
      self.redraw()

    # Outliers
    elif event.key == OUTL:
      self.Transits.set_active(False)
      self.Zoom.set_active(False)
      self.Split.set_active(False)
      self.Outliers.set_active(not self.Outliers.active)
      self.redraw()
      
    # Splits
    elif event.key == SPLT:
      self.Transits.set_active(False)
      self.Zoom.set_active(False)
      self.Outliers.set_active(False)
      self.Split.set_active(not self.Split.active)
      self.redraw()
    
    # Toggle pixels
    if event.key == PIXL:
      if self.state[0] == 'fsum': self.state[0] = 'fpix'
      elif self.state[0] == 'fpix': self.state[0] = 'fsum'
      self.redraw(preserve_lims = False)
    
    # Toggle errorbars
    if event.key == ERRB:
      if self.state[1] == 'none': self.state[1] = 'errorbars'
      elif self.state[1] == 'errorbars': self.state[1] = 'none'
      self.redraw()
      
    # Reset
    elif event.key == RSET:
      self.info = "reset"
      pl.close()

    # Next
    elif event.key == NEXT:
      self.info = "next"
      pl.close()
    
    # Previous
    elif event.key == PREV:
      self.info = "prev"
      pl.close()
    
    # Quit
    elif event.key == QUIT:
      self.info = "quit"
      pl.close()
    
    # Padding
    elif event.key == TPAD:
      self.info = "tpad"
      pl.close()
    
    elif event.key == BLND:
      self.info = "blind"
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
    if (event.inaxes is None) or (self.hide) or (not self.Transits.active) or (event.button != 1): 
      return
    x, y = event.xdata, event.ydata
    i = np.argmin(np.abs(x - self.cad))
    j = np.argmin(np.abs(self.cad[i] - self.tNc))
    self.tNc[j] = x

    # Update
    self.UpdateTransits()
  
  def UpdateTransits(self):
    '''
    
    '''
    
    # Remove these from the plot
    for l in self._transit_lines:
      for i in l:
        if len(i): i.pop(0).remove()
    self._transit_lines = []
    
    # Figure out the transit indices
    self._transits_narrow = []
    self._transits_wide = []
    self._transits_wide_tag = []
    for j, tc in enumerate(self.tNc):
      i = np.where(np.abs(self.cad - tc) <= self.cptbkg / 2.)[0]
      self._transits_narrow.extend(i)
      i = np.where(np.abs(self.cad - tc) <= self.cpttrn / 2.)[0]
      self._transits_wide.extend(i)
      self._transits_wide_tag.extend([j for k in i])
    
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
  def transits_wide_tag(self):
    return np.array(sorted(self._transits_wide_tag), dtype = int)
  
  @property
  def transits_wide(self):
    return np.array(sorted(self._transits_wide), dtype = int)
  
  @property
  def transits_narrow(self):
    return np.array(sorted(self._transits_narrow), dtype = int)
  
  @property
  def jumps(self):
    return np.array(sorted(self._jumps), dtype = int)
  
def Preprocess(input_file = None):
  '''

  '''

  # Load inputs
  inp = Input(input_file)

  # Load MPL backend
  import matplotlib
  if inp.interactive:
    # Are we running this interactively? If so, use TkAgg
    matplotlib.use('TkAgg', warn = False, force = True)
    if matplotlib.get_backend() != 'TkAgg':
      print("WARNING: Unable to load TkAgg backend. Interactive mode disabled.")
      inp.interactive = False
  else:
    # Let's try to use the Agg backend. Not a big deal if it doesn't work
    import matplotlib
    matplotlib.use('Agg', warn = False)
  import matplotlib.pyplot as pl

  # Check for a saved version
  if not inp.clobber:
  
    try:
    
      # Try to load it
      GetData(inp.id, data_type = 'bkg', datadir = inp.datadir)
      if not inp.quiet: 
        print("Loading saved data...")
      return True
    except:
    
      # The file doesn't exist
      pass
  
  # Grab the data
  if not inp.quiet: 
    print("Downloading target data...")
  data = DownloadData(inp.id, inp.dataset, long_cadence = inp.long_cadence, 
                      clobber = inp.clobber, datadir = inp.datadir, 
                      bad_bits = inp.bad_bits, aperture = inp.aperture, 
                      quarters = inp.quarters, quiet = inp.quiet, pskwargs = 
                      inp.pskwargs)
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir, 
                      clobber = inp.clobber, ttvs = inp.ttvs, pad = inp.padbkg,
                      pskwargs = inp.pskwargs, trninfo = inp.trninfo)
  tN = info['tN']
  tdur = info['tdur']
  per = info['per']
  hash = info['hash']
  data_new = EmptyData(inp.quarters)
  data_trn = EmptyData(inp.quarters)
  data_bkg = EmptyData(inp.quarters)

  # Loop over all quarters
  if not inp.quiet: 
    print("Inspecting...")
  uo = {}; [uo.update({q: []}) for q in inp.quarters]
  uj = {}; [uj.update({q: []}) for q in inp.quarters]
  utn = {}; [utn.update({q: []}) for q in inp.quarters]
  utw = {}; [utw.update({q: []}) for q in inp.quarters]
  q = inp.quarters[0]
  dq = 1
  cpttrn = None
  cptbkg = None

  while q in inp.quarters:

    # Empty quarter?
    if data[q]['time'] == []:
      q += dq
      continue

    # Gap tolerance in cadences  
    tpc = np.median(data[q]['time'][1:] - data[q]['time'][:-1])
    cad_tol = inp.dt_tol / tpc
  
    # Cadences per transit
    if cptbkg is None:
      cptbkg = tdur * inp.padbkg / tpc
    if cpttrn is None:
      cpttrn = tdur * inp.padtrn / tpc
  
    # Disable toolbar and shortcuts
    if inp.interactive:
      orig = DisableShortcuts()

    # Set up the plot
    fig, ax = pl.subplots(1, 1, figsize = (16, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left = 0.075, right = 0.95)   
    sel = Viewer(fig, ax, inp.id, q, data[q], tN, cptbkg, cpttrn, inp.datadir,
                 split_cads = inp.split_cads, cad_tol = cad_tol, 
                 min_sz = inp.min_sz, interactive = inp.interactive)
            
    # If user is re-visiting this quarter, update with their selections 
    if len(uj[q]): 
      sel._jumps = list(uj[q])
    if len(uo[q]): 
      sel._outliers = list(uo[q])
    if len(utn[q]): 
      sel._transits_narrow = list(utn[q])
    if len(utw[q]): 
      sel._transits_wide = list(utw[q])
    
    if inp.interactive:
      fig.canvas.set_window_title('KEPLER %.2f: Quarter %02d' % (inp.id, q)) 
    
    sel.UpdateTransits()
    sel.redraw()

    # Bring window to the front and fullscreen it
    if inp.interactive:
      fig.canvas.manager.window.attributes('-topmost', 1)
      if inp.fullscreen:
        fig.canvas.manager.window.attributes('-fullscreen', 1) 
    
    # Save the figure
    fig.savefig(os.path.join(inp.datadir, str(inp.id), '_plots', "Q%02d.png" % q), bbox_inches = 'tight')
    
    # Show the figure
    if inp.interactive:
      pl.show()
    
    pl.close()

    # What will we do next time?
    if sel.info == "next":
      dq = 1
    elif sel.info == "prev":
      dq = -1
    elif sel.info == "reset":
      continue
    elif sel.info == "help":
      ShowHelp()
      # Save user selections
      uo[q] = sel.outliers
      uj[q] = sel.jumps
      utn[q] = sel.transits_narrow
      utw[q] = sel.transits_wide
      tN = sel.tN
      cptbkg = sel.cptbkg
      cpttrn = sel.cpttrn
      # Re-plot
      continue
    elif sel.info == "tpad":
      try:
        cptbkg = float(prompt("Cadences per transit (background) [%.2f]: " % sel.cptbkg))
        if cptbkg <= 0:
          raise Exception("")
        sel.cptbkg = cptbkg
      except:
        pass
      try:
        cpttrn = float(prompt("Cadences per transit (transits) [%.2f]: " % sel.cpttrn))
        if cpttrn <= 0:
          raise Exception("")
        sel.cpttrn = cpttrn
      except:
        pass
      # Save user selections
      uo[q] = sel.outliers
      uj[q] = sel.jumps
      utn[q] = sel.transits_narrow
      utw[q] = sel.transits_wide
      tN = sel.tN
      cptbkg = sel.cptbkg
      cpttrn = sel.cpttrn
      # Re-plot
      continue
    elif sel.info == "quit":
      EnableShortcuts()
      return False
    elif sel.info == "blind":
      EnableShortcuts()
      inp.interactive = False
    else:
      EnableShortcuts()
      return False
  
    jumps = sel.jumps
    transits_narrow = sel.transits_narrow
    transits_wide = sel.transits_wide
    transits_wide_tag = sel.transits_wide_tag
    outliers = sel.outliers
    tN = sel.tN
    cptbkg = sel.cptbkg
    cpttrn = sel.cpttrn
  
    # Store the user-defined outliers, jumps, and transits
    uo[q] = outliers
    uj[q] = jumps
    utn[q] = transits_narrow
    utw[q] = transits_wide

    # Split the data
    j = np.concatenate([[0], jumps, [len(data[q]['time']) - 1]])
    for arr in ['time', 'fpix', 'perr', 'cadn', 'pdcf', 'pdce']:

      # All data and background data
      for a, b in zip(j[:-1], j[1:]):
        ai = [i for i in range(a, b) if i not in outliers]
        bi = [i for i in range(a, b) if i not in np.append(outliers, transits_narrow)]
        data_new[q][arr].append(np.array(data[q][arr][ai]))
        data_bkg[q][arr].append(np.array(data[q][arr][bi]))
    
      # Transit-only data        
      for i in range(len(tN)):
        ti = transits_wide[np.where(transits_wide_tag == i)]
        ti = [foo for foo in ti if foo not in outliers]
      
        # We're discarding transits that span two chunks
        if len(list(set(jumps).intersection(ti))) == 0 and len(ti) > 0:
          data_trn[q][arr].append(data[q][arr][ti])
          data_trn[q]['trni'].append(i)
    
    # Add the crowding
    data_new[q]['crwd'] = data[q]['crwd']
    data_trn[q]['crwd'] = data[q]['crwd']
    data_bkg[q]['crwd'] = data[q]['crwd']
    
    # Increment and loop
    q += dq

  # Save the data
  if not inp.quiet: print("Saving data to disk...")
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'prc.npz'), data = data_new, hash = GitHash())
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'trn.npz'), data = data_trn, hash = GitHash())
  np.savez_compressed(os.path.join(inp.datadir, str(inp.id), '_data', 'bkg.npz'), data = data_bkg, hash = GitHash())

  # Re-enable toolbar and shortcuts
  if inp.interactive:
    EnableShortcuts()

  return True