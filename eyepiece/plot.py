#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
plot.py
-------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from .download import DownloadInfo
from .utils import Input, GetData
from .detrend import GetWhitenedData, GetBadChunks
import matplotlib.pyplot as pl
import numpy as np
import os

def PlotDetrended(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  
  if not inp.quiet:
    print("Plotting detrended background flux...")
  
  # Load some info
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']
  tdur = info['tdur']
  
  # Index of first PLD coefficient in ``x``
  iPLD = len(inp.kernel.pars)
  
  # Plot the decorrelated data
  fig = pl.figure(figsize = (48, 24))
  #fig.subplots_adjust(hspace = 10.)  
  
  ax = [pl.subplot2grid((48,7), (0,0), colspan=7, rowspan=12),
        pl.subplot2grid((48,7), (14,0), colspan=7, rowspan=12),
        pl.subplot2grid((48,7), (28,0), colspan=7, rowspan=12)]
  
  axzoom = [pl.subplot2grid((48,7), (44,0), rowspan=4),
            pl.subplot2grid((48,7), (44,1), rowspan=4),
            pl.subplot2grid((48,7), (44,2), rowspan=4),
            pl.subplot2grid((48,7), (44,3), rowspan=4),
            pl.subplot2grid((48,7), (44,4), rowspan=4),
            pl.subplot2grid((48,7), (44,5), rowspan=4),
            pl.subplot2grid((48,7), (44,6), rowspan=4)]
  
  # Some miscellaneous info
  lt = [None] * (inp.quarters[-1] + 1)
  wf = [""] * (inp.quarters[-1] + 1)
  fc = np.zeros(inp.quarters[-1] + 1)
  ni = np.zeros(inp.quarters[-1] + 1)
  ll = np.zeros(inp.quarters[-1] + 1)
  cc = [None] * (inp.quarters[-1] + 1)
  
  FLUX = np.array([], dtype = float)
  TIME = np.array([], dtype = float)
  
  # Loop over all quarters
  for q in inp.quarters:
    
    # Load the decorrelated data
    files = [os.path.join(detpath, f) for f in os.listdir(detpath) 
             if f.startswith('%02d.' % q) and f.endswith('.npz')]
    
    # Is there data this quarter?
    if len(files) == 0:
      continue

    # Grab the highest likelihood run
    lnl = np.zeros_like(files, dtype = float)
    for i, f in enumerate(files):
      lnl[i] = float(np.load(f)['lnlike'])
    res = np.load(files[np.argmax(lnl)])
  
    # Grab the detrending info
    time = res['time']
    fsum = res['fsum']
    ypld = res['ypld']
    gpmu = res['gpmu']
    gperr = res['gperr']
    x = res['x']
    lnlike = res['lnlike']
    info = res['info'][()]
    init = res['init']
    
    # The SAP flux (median-subtracted)
    ax[0].plot(time, fsum - np.nanmedian(fsum), 'k.', alpha = 0.3)
    
    # The PLD-detrended SAP flux (blue) and the GP (red)
    ax[1].plot(time, ypld - np.nanmedian(ypld), 'b.', alpha = 0.3)
    ax[1].plot(time, gpmu - np.nanmedian(ypld), 'r-')
  
    # The fully detrended flux
    f = ypld - gpmu
    ax[2].plot(time, f, 'b.', alpha = 0.3)
    
    # Running arrays
    FLUX = np.append(FLUX, f)
    TIME = np.append(TIME, time)
    
    # Appearance
    [axis.set_xticklabels([]) for axis in ax[:-1]]
    [axis.margins(0, 0.01) for axis in ax]
    
    # Extra info
    lt[q] = time[-1]
    fc[q] = info['funcalls']
    ni[q] = info['nit']
    if info['warnflag']:
      wf[q] = str(info['task'])
      if 'ABNORMAL_TERMINATION_IN_LNSRCH' in wf[q]:
        wf[q] = 'AT_LNSRCH'
    ll[q] = lnlike
    cc[q] = x
      
  ltq = ax[0].get_xlim()[0]  
  yp0 = ax[0].get_ylim()[1]
  yp1 = ax[1].get_ylim()[1]
  yp2 = ax[2].get_ylim()[1]
  
  for q in inp.quarters:
    
    # This stores the last timestamp of the quarter
    if lt[q] is None:
      continue
    
    # If the data spans more than 20 days, plot some info (otherwise, it won't fit!)
    if lt[q] - ltq > 20:
    
      # Quarter number
      ax[0].annotate(q, ((ltq + lt[q]) / 2., yp0), ha='center', va='bottom', fontsize = 24)
    
      if inp.plot_det_info:
        # Best coeff values
        ax[0].annotate("\n   PLD COEFFS", (ltq, yp0), ha='left', va='top', fontsize = 8, color = 'r')
        for i, c in enumerate(cc[q][iPLD:]):
          ax[0].annotate("\n" * (i + 2) + "   %.1f" % c, (ltq, yp0), ha='left', va='top', fontsize = 8, color = 'r')
    
        # Best GP param values
        ax[1].annotate("\n   GP PARAMS", (ltq, yp1), ha='left', va='top', fontsize = 8)
        for i, c in enumerate(cc[q][:iPLD]):
          ax[1].annotate("\n" * (i + 2) + "   %.2f" % c, (ltq, yp1), ha='left', va='top', fontsize = 8)
      
        # Optimization info
        if wf[q] == "":
          ax[2].annotate("\n   SUCCESS", (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'b')
        else:
          ax[2].annotate("\n   %s" % wf[q], (ltq, yp2), ha='left', va='top', fontsize = 8, color = 'r')
        if inp.kernel is not None:
          ax[2].annotate("\n\n   CALLS: %d" % fc[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
          ax[2].annotate("\n\n\n   NITR: %d" % ni[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
          ax[2].annotate("\n\n\n\n   LNLK: %.2f" % ll[q], (ltq, yp2), ha='left', va='top', fontsize = 8)
    
    for axis in ax:
      axis.axvline(lt[q], color='k', ls = '--')
    
    ltq = lt[q]
  
  # Let's identify potentially bad parts of the detrended data
  chunks = GetBadChunks(FLUX, sig_tol = 3.)
  [ax[2].plot(TIME[chunk], FLUX[chunk], 'r.') for chunk in chunks]
  
  # Plot the bad chunks as insets
  for i, axz, chunk in zip(range(len(axzoom)), axzoom, chunks):

    # Expand around them a bit
    a, b = chunk[0] - 3 * len(chunk), chunk[-1] + 3 * len(chunk)
    axz.plot(TIME[a:b], FLUX[a:b], 'b.')
    axz.plot(TIME[chunk], FLUX[chunk], 'r.')
    
    # Appearance
    axz.set_title('Bad Chunk #%d' % (i + 1), fontsize = 22, fontweight = 'bold', y = 1.025)
    axz.set_axis_bgcolor((0.95, 0.95, 0.95))
  
  # Hide empty plots
  if len(chunks) < len(axzoom):
    for axz in axzoom[len(chunks):]:
      axz.set_visible(False)
  
  # Labels and titles
  ax[0].set_title('Raw Background Flux', fontsize = 28, fontweight = 'bold', y = 1.1) 
  ax[1].set_title('PLD-Decorrelated Flux', fontsize = 28, fontweight = 'bold', y = 1.025)  
  ax[2].set_title('PLD+GP-Decorrelated Flux', fontsize = 28, fontweight = 'bold', y = 1.025) 
  ax[-1].set_xlabel('Time (Days)', fontsize = 24)
  [axis.set_ylabel(r'$\Delta$ Counts', fontsize = 24) for axis in ax]
  
  # Appearance
  for s in ['top', 'bottom', 'left', 'right']:
    [axis.spines[s].set_linewidth(2) for axis in ax + axzoom]
  [tick.label.set_fontsize(20) for tick in ax[-1].xaxis.get_major_ticks()]
  [tick.label.set_fontsize(18) for axis in ax for tick in axis.yaxis.get_major_ticks()] 
  ax[0].set_axis_bgcolor((1.0, 0.95, 0.95))
  ax[1].set_axis_bgcolor((0.95, 0.95, 0.95))
  ax[2].set_axis_bgcolor((0.95, 0.95, 1.0))
  
  # Save and return
  fig.savefig(os.path.join(inp.datadir, str(inp.id), '_plots', 'detrended.png'), bbox_inches = 'tight')
  
  return fig, ax

def PlotTransits(input_file = None):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')

  if not inp.quiet:
    print("Plotting transits...")

  # Load the info
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir); info.update(inp.info)
  tdur = info['tdur']
  t, f, e = GetWhitenedData(input_file, folded = True)

  # Plot
  fig, ax = pl.subplots(1, 1, figsize = (14, 6))
  xlim = (-inp.padtrn * tdur / 2., inp.padtrn * tdur / 2.)
  fvis = f[np.where((t > xlim[0]) & (t < xlim[1]))]
  minf = np.min(fvis)
  maxf = np.max(fvis)
  padf = 0.1 * (maxf - minf)
  ylim = (minf - padf, maxf + padf)
  ax.plot(t, f, 'k.', alpha = min(1.0, max(0.05, 375. / len(fvis))))
  
  # Bin to median
  bins = np.linspace(xlim[0], xlim[1], inp.tbins)
  delta = bins[1] - bins[0]
  idx  = np.digitize(t, bins)
  med = [np.median(f[idx == k]) for k in range(inp.tbins)]
  ax.plot(bins - delta / 2., med, 'ro', alpha = 0.75)
  
  ax.set_xlim(xlim)  
  ax.set_ylim(ylim)
  ax.set_title('Folded Whitened Transits', fontsize = 24)
  ax.set_xlabel('Time (days)', fontsize = 22)
  ax.set_ylabel('Flux', fontsize = 22)
  
  # Save and return
  fig.savefig(os.path.join(inp.datadir, str(inp.id), '_plots', 'folded.png'), bbox_inches = 'tight')
  
  return fig, ax