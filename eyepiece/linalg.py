#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
lnlike.py
---------

'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import george

__all__ = ['LnLike', 'Whiten']

def LnLike(x, time, fpix, perr, fsum = None, tmod = None, lndet = True, 
           predict = False, kernel = 1. * george.kernels.Matern32Kernel(1.)):
  '''
  Returns the *Ln(Likelihood)* of the model given an array of parameters ``x``,
  as well as its gradient (computed analytically) with respect to ``x``.
  
  :param x: The array of model parameters, starting with \
            the GP kernel parameters and followed by the \
            PLD pixel coefficients.
  :type x: :class:``numpy.ndarray``
  
  :param time: The array of times at which the data was taken
  :type time: :class:``numpy.ndarray``
  
  :param fpix: An array of shape ``(N, NP)`` of the individual pixel fluxes, where \
               ``N`` is the number of data points and ``NP`` is the number of pixels \
               in the aperture
  :type fpix: :class:``numpy.ndarray``
  
  :param perr: An array of shape ``(N, NP)`` of the errors on the pixel fluxes
  :type perr: :class:``numpy.ndarray``
  
  :param fsum: An array of length ``N`` corresponding to the total flux at each time \
               point. If not provided, this is computed by summing over the second \
               axis of ``fpix``.
  :type fsum: :class:``numpy.ndarray``, optional
  
  :param tmod: An array of length ``N`` corresponding to the transit model. If not \
               provided, defaults to ``np.ones(N)`` (i.e., no transit)
  :type tmod: :class:``numpy.ndarray``, optional
  
  :param lndet: Whether or not to compute the contribution of the *ln|K|* term to \
                the gradient. This contribution is usually small and may be \
                neglected in some cases to speed up the computation. Default is \
                ``True``
  :type fpix: ``bool``, optional
  
  :param predict: If ``True``, also returns the PLD-detrended flux ``y``, the GP \
                  mean flux ``mu`` and the GP errors ``gperr`` for the \
                  GP prediction given the set of parameters ``x``. Default ``False``
  :type predict: ``bool``, optional
  
  :param kernel: The kernel to use in the GP computation. Default is a *Matern 3/2* \
                 kernel
  :type fpix: :class:``george.kernels.Kernel``
  
  '''
  
  # Calculate fsum if not provided
  if fsum is None:
    fsum = np.sum(fpix, axis = 1)
  
  # PLD coefficients
  c = x[len(kernel.pars):]
  
  # Kernel params
  kernel.pars = x[:len(kernel.pars)]
  
  # Number of data points
  K = len(time)
  
  # Inverse of the transit model
  if tmod is None:
    D = np.ones_like(time)
  else:
    D = 1. / tmod
  
  # The pixel model
  pmod = np.sum(fpix * np.outer(1. / fsum, c), axis = 1)
  
  # The PLD-detrended data
  y = fsum - pmod

  # Errors on detrended data (y)
  X = D + pmod / fsum
  B = X.reshape(K, 1) * perr - c * perr / fsum.reshape(K, 1)
  yerr = np.sum(B ** 2, axis = 1) ** 0.5

  # Compute the likelihood
  gp = george.GP(kernel)
  try:
    gp.compute(time, yerr)
    ll = gp.lnlikelihood(y)
  except:
    # If something went wrong, return zero likelihood and zero gradient
    if predict:
      return (-np.inf, np.zeros_like(x), y, np.zeros_like(y), np.zeros_like(y))
    else:
      return (-np.inf, np.zeros_like(x))

  # Compute the gradient of the likelihood with respect to the PLD coefficients
  # First, the gradient assuming the covariance is independent of the coeffs
  A = gp.solver.apply_inverse(y)
  R = fpix / fsum.reshape(K, 1)
  grad_pld = np.dot(R.T, A)
  
  # Pre-compute some stuff
  H = np.array([perr[k] * (X[k] - (c / fsum[k])) for k in range(K)])
  J = np.array([perr[k] / fsum[k] for k in range(K)])
  
  # Are we computing the very small contribution from the ln-det term?
  # If so, pre-compute the diagonal of the inverse covariance matrix
  if lndet:
    KID = np.diag(gp.solver.apply_inverse(np.eye(K)))
  
  # Now add the covariance term  
  for m, _ in enumerate(grad_pld):

    # The Kronecker delta
    KD = np.zeros_like(grad_pld); KD[m] = 1
    
    # Derivative of sigma^2 with respect to the mth PLD coefficient
    dS2dC = 2 * np.diag(np.sum(H * (J * R[:,m].reshape(K,1) - np.outer(J[:,m], KD)), axis = 1))
    
    # The covariance term
    grad_pld[m] += 0.5 * np.dot(A.T, np.dot(dS2dC, A))
    
    # The small ln(det) term
    if lndet:
      grad_pld[m] -= np.dot(KID, np.diag(dS2dC))
  
  # Append the PLD gradient to the GP gradient
  # Note that george returns dLnLike/dLnX, so we divide by X to get dLnLike/dX.
  grad_ll = np.append(gp.grad_lnlikelihood(y) / kernel.pars, grad_pld)
  
  # Should we return a sample prediction?
  if predict:
    mu, cov = gp.predict(y, time)
    gperr = np.sqrt(np.diag(cov)); del cov
    return (ll, grad_ll, y, mu, gperr)
  else:
    return (ll, grad_ll)

def Whiten(x, b_time, b_fpix, b_perr, time, fpix, kernel = 1. * george.kernels.Matern32Kernel(1.), crowding = None):
  '''
  
  '''
  
  # Calculate fsum
  b_fsum = np.sum(b_fpix, axis = 1)
  fsum = np.sum(fpix, axis = 1)
  
  # PLD coefficients
  c = x[len(kernel.pars):]
  
  # Kernel params
  kernel.pars = x[:len(kernel.pars)]
  
  # Number of background data points
  b_K = len(b_time)

  # The pixel model
  b_pmod = np.sum(b_fpix * np.outer(1. / b_fsum, c), axis = 1)
  pmod = np.sum(fpix * np.outer(1. / fsum, c), axis = 1)

  # Errors on detrended background data
  X = 1. + b_pmod / b_fsum
  B = X.reshape(b_K, 1) * b_perr - c * b_perr / b_fsum.reshape(b_K, 1)
  b_yerr = np.sum(B ** 2, axis = 1) ** 0.5

  # Compute the likelihood
  gp = george.GP(kernel)
  gp.compute(b_time, b_yerr)
  mu, _ = gp.predict(b_fsum - b_pmod, time)
  
  # The full decorrelated flux with baseline = 1.
  dflux = 1. + (fsum - mu - pmod) / fsum
  
  # Correct for crowding?
  if crowding is not None:
    dflux = (dflux - 1.) / crowding + 1.
  
  return dflux