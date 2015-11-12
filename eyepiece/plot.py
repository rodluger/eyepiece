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
import george

# Python 2/3 compatibility
try:
  FileNotFoundError
except:
  FileNotFoundError = IOError

def PlotDetrended(input_file = None):
  '''
  
  '''
  
  # Load inputs
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')
  
  # Have we done this already?
  if not inp.clobber:
    if os.path.exists(os.path.join(inp.datadir, str(inp.id), '_plots', 'detrended.png')):
      return None, None
  
  if not inp.quiet:
    print("Plotting detrended flux...")
  
  # Load some info
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir); info.update(inp.info)
  tN = info['tN']
  tdur = info['tdur']
  
  # The full processed data
  prc = GetData(inp.id, data_type = 'prc', datadir = inp.datadir)
  
  # Index of first PLD coefficient in ``x``
  iPLD = len(inp.kernel.pars)
  
  # Plot the decorrelated data
  fig = pl.figure(figsize = (48, 24))  
  
  ax = [pl.subplot2grid((48,7), (0,0), colspan=7, rowspan=12),
        pl.subplot2grid((48,7), (14,0), colspan=7, rowspan=12),
        pl.subplot2grid((48,7), (28,0), colspan=7, rowspan=12)]
  
  axzoom = [pl.subplot2grid((48,7), (43,0), rowspan=5),
            pl.subplot2grid((48,7), (43,1), rowspan=5),
            pl.subplot2grid((48,7), (43,2), rowspan=5),
            pl.subplot2grid((48,7), (43,3), rowspan=5),
            pl.subplot2grid((48,7), (43,4), rowspan=5)]
  
  axfold = pl.subplot2grid((48,7), (43,5), rowspan=5, colspan=2)
  
  # Some miscellaneous info
  lt = [None] * (inp.quarters[-1] + 1)
  wf = [""] * (inp.quarters[-1] + 1)
  fc = np.zeros(inp.quarters[-1] + 1)
  ni = np.zeros(inp.quarters[-1] + 1)
  ll = np.zeros(inp.quarters[-1] + 1)
  cc = [None] * (inp.quarters[-1] + 1)
  
  # Cumulative arrays
  FLUX = np.array([], dtype = float)
  TIME = np.array([], dtype = float)
  
  # Loop over all quarters
  for q in inp.quarters:
    
    # Empty quarter?
    if len(prc[q]['time']) == 0:
      continue
    
    # Get detrending info
    info = prc[q]['info']
    lnlike = prc[q]['lnlike']
    dvec = prc[q]['dvec']
    inp.kernel.pars = dvec[:iPLD]
    gp = george.GP(inp.kernel)
    
    time = np.array([], dtype = float)
    fsum = np.array([], dtype = float)
    ypld = np.array([], dtype = float)
    gpmu = np.array([], dtype = float)
    for t_, p_, y_, e_ in zip(prc[q]['time'], prc[q]['fpix'], prc[q]['ypld'], prc[q]['yerr']):
      
      time = np.append(time, t_)
      fsum = np.append(fsum, np.sum(p_, axis = 1))
      ypld = np.append(ypld, y_)
      
      gp.compute(t_, e_)
      mu, cov = gp.predict(y_, t_); del cov
      gpmu = np.append(gpmu, mu)
      
    # The SAP flux (median-subtracted), with transits
    ax[0].plot(time, fsum - np.nanmedian(fsum), 'k.', alpha = 0.3)
    
    # The PLD-detrended SAP flux (blue) and the GP (red)
    ax[1].plot(time, ypld - np.nanmedian(ypld), 'b.', alpha = 0.3)
    ax[1].plot(time, gpmu - np.nanmedian(ypld), 'r-')
    
    # The fully detrended flux
    f = ypld - gpmu
    ax[2].plot(time, f, 'b.', alpha = 0.3)
    
    # Cumulative arrays
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
    cc[q] = dvec
      
  ltq = ax[0].get_xlim()[0]  
  yp0 = ax[0].get_ylim()[1]
  yp1 = ax[1].get_ylim()[1]
  yb2 = ax[2].get_ylim()[0]
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
  chunks = GetBadChunks(FLUX, sig_tol = 3., sort = True)
  [ax[2].plot(TIME[chunk], FLUX[chunk], 'r.') for chunk in chunks]
  [ax[2].annotate(i, (np.median(TIME[chunk]), yb2), ha = 'left', va = 'bottom', fontsize = 8, fontweight = 'bold', color = 'r') for i,chunk in enumerate(chunks)]
  
  # Plot the bad chunks as insets
  for i, axz, chunk in zip(range(len(axzoom)), axzoom, chunks):

    # Expand around them a bit
    a, b = chunk[0] - 3 * len(chunk), chunk[-1] + 3 * len(chunk)
    axz.plot(TIME[a:b], FLUX[a:b], 'b.')
    axz.plot(TIME[chunk], FLUX[chunk], 'r.')
    
    # Are there transits nearby?
    idx = np.where((tN > TIME[a]) & (tN < TIME[b]))[0]
    for j in idx:
      axz.axvline(tN[j], color = 'r', alpha = 0.5)
    
    # Appearance
    axz.set_title('Bad Chunk #%d' % (i + 1), fontsize = 22, fontweight = 'bold', y = 1.025)
    axz.set_axis_bgcolor((0.95, 0.95, 0.95))
  
  # Hide empty plots
  if len(chunks) < len(axzoom):
    for axz in axzoom[len(chunks):]:
      axz.set_visible(False)
  
  # Plot the folded transits
  if type(inp.id) is float:
    axfold = PlotTransits(input_file, ax = axfold)
    axfold.set_title('Folded Whitened Transits: KOI %.2f' % inp.id, fontsize = 22, fontweight = 'bold', y = 1.025)
  else:
    axfold.set_visible(False)
  
  # Labels and titles
  ax[0].set_title('Raw Median-Subtracted Flux', fontsize = 28, fontweight = 'bold', y = 1.1) 
  ax[1].set_title('PLD-Decorrelated Background Flux', fontsize = 28, fontweight = 'bold', y = 1.025)  
  ax[2].set_title('PLD+GP-Decorrelated Background Flux', fontsize = 28, fontweight = 'bold', y = 1.025) 
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

def PlotTransits(input_file = None, ax = None):
  '''
  
  '''
  
  # Input file
  inp = Input(input_file)
  detpath = os.path.join(inp.datadir, str(inp.id), '_detrend')

  # Have we done this already?
  if ax is None and not inp.clobber:
    if os.path.exists(os.path.join(inp.datadir, str(inp.id), '_plots', 'folded.png')):
      return None, None

  if ax is None and not inp.quiet:
    print("Plotting transits...")

  # Load the info
  info = DownloadInfo(inp.id, inp.dataset, datadir = inp.datadir); info.update(inp.info)
  tdur = info['tdur']
  
  # Load the whitened data
  try:
    foo = np.load(os.path.join(inp.datadir, str(inp.id), '_data', 'white.npz'))
  except FileNotFoundError:
    raise FileNotFoundError("No detrending information found for this target.")
    
  t = foo['t']
  f = foo['f']
  e = foo['e']

  # Plot
  if ax is None:
    userax = False
    fig, ax = pl.subplots(1, 1, figsize = (14, 6))
  else:
    userax = True
    
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
  
  # Plot the quick-and-dirty transit model (if available)
  try:
    foo = np.load(os.path.join(inp.datadir, str(inp.id), '_data', 'tmod.npz'))
    t = foo['t']
    tmod = foo['tmod']
    ax.plot(t, tmod, 'r-', alpha = 0.5)
  except FileNotFoundError:
    pass
    
  ax.set_xlim(xlim)  
  ax.set_ylim(ylim)
  
  if not userax:
    ax.set_title('Folded Whitened Transits', fontsize = 24)
    ax.set_xlabel('Time (days)', fontsize = 22)
    ax.set_ylabel('Flux', fontsize = 22)
  
  if not userax:
    fig.savefig(os.path.join(inp.datadir, str(inp.id), '_plots', 'folded.png'), bbox_inches = 'tight')
    return fig, ax
  else:
    return ax