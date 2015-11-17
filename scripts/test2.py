import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
from eyepiece.utils import GetData
from eyepiece.download import DownloadInfo
import george
import numpy as np

prc = GetData(1612.01, data_type = 'prc', datadir = '/gscratch/vsm/rodluger/lightcurves')
info = DownloadInfo(1612.01, 'Kepler', datadir = '/gscratch/vsm/rodluger/lightcurves')
tN = info['tN']
iPLD = 2
q = 2
 
kernel = 1. * george.kernels.Matern32Kernel(1.)
kernel.pars = prc[q]['dvec'][:iPLD]
gp = george.GP(kernel)

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
  
pl.plot(time, ypld - gpmu, 'b.', alpha = 0.3)


for ti in tN:
  pl.axvline(ti, color = 'r', alpha = 0.25)
  
pl.show()