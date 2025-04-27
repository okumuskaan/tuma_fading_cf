"""
Helper Functions for Decoder Setup

This file contains functions for setting priors, generating covariance matrices,
and creating grid structures used in the centralized decoding of the TUMA framework.

Functions:
- compute_prior: Compute prior probabilities for message multiplicities.
- generate_priors: Generate priors for all zones.
- generate_cov_matrix: Generate covariance matrices based on user and AP positions.
- generate_uniform_grid: Create uniform grids of quantized points for zones.
- generate_all_covs: Generate covariance matrices for all multiplicities via sampling.

Usage:
This module is intended for use in the TUMA centralized decoder setup and can be imported as:
    from utils.helper_decoder import compute_prior, generate_priors

Author: Kaan Okumus
Date: April 2025
"""

import numpy as np
import scipy.stats

from utils.helper_cf_tuma_tx import gamma

def compute_prior(Ka, Ma, M, uniform=False, Kmax=None):
    """
    Compute prior distribution for message multiplicities.

    Parameters:
    Ka : int
        Total number of transmitted messages.
    Ma : int
        Number of users per zone.
    M : int
        Total number of quantized message positions.
    uniform : bool, optional
        If True, assigns a uniform prior over multiplicities (default: False).
    Kmax : int, optional
        Maximum multiplicity value to consider (default: None).

    Returns:
    np.ndarray
        Array representing the prior distribution for multiplicities.
    """
    bin_pmfs = scipy.stats.binom.pmf(np.arange(1, Ka + 1), Ka, 1 / Ma)
    bin_pmfs /= bin_pmfs.sum()
    prior = np.zeros((Ka + 1, 1))
    p0 = 1 - Ma / M
    prior[0, 0] = p0
    if uniform:
        prior[1:, 0] = (1 - p0) / (Kmax)
    else:
        prior[1:, 0] = (1 - p0) * bin_pmfs
    return prior

def generate_priors(Kaus, Maus, Mus, Kmax):
    """
    Generate priors for all zones.

    Parameters:
    Kaus : list of int
        List of total transmitted messages per zone.
    Maus : list of int
        List of users per zone.
    Mus : list of int
        List of quantized message positions per zone.
    Kmax : int
        Maximum multiplicity (number of sources per message).

    Returns:
    list of np.ndarray
        List of prior distributions, one per zone.
    """
    U = len(Kaus)
    priors = [
        compute_prior(Kaus[u], Maus[u], Mus[u])[: Kmax + 1].reshape(-1) for u in range(U)
    ]
    priors = [prior / prior.sum() for prior in priors]
    priors = [prior.reshape(1, -1) * np.ones((Mus[0], 1)) for prior in priors]
    return priors

def generate_cov_matrix(pos, nus, A, rho=3.67, d0=0.01357):
    """
    Generate covariance matrix based on user positions and AP locations.

    Parameters:
    pos : np.ndarray
        User positions (complex coordinates).
    nus : np.ndarray
        AP positions (complex coordinates).
    A : int
        Number of antennas per AP.
    rho : float, optional
        Path loss exponent (default: 3.67).
    d0 : float, optional
        Reference distance for path loss (default: 0.01357).

    Returns:
    np.ndarray
        Covariance matrices corresponding to input positions.
    """
    pos_shape = list(pos.shape)
    return (gamma(np.expand_dims(pos,axis=(-1,-2)), nus.reshape(-1,1), rho=rho, d0=d0)*np.ones((1,A))).reshape(pos_shape + [-1])

def generate_uniform_grid(side, num_points, zone_centers, margin=0):
    """
    Generate uniform grid of points for all zones.

    Parameters:
    side : float
        Side length of each zone.
    num_points : int
        Total number of points to generate.
    zone_centers : np.ndarray
        Centers of zones (complex coordinates).
    margin : float, optional
        Margin from the edges (default: 0).

    Returns:
    np.ndarray
        Grid points for each zone.
    """
    # Calculate approximate number of points per axis
    num_per_axis = int(np.sqrt(num_points))

    # Generate grid points
    x = np.linspace(-side/2 + margin, side/2 - margin, num_per_axis)
    y = np.linspace(-side/2 + margin, side/2 - margin, num_per_axis)
    xx, yy = np.meshgrid(x, y)
    base_qs = xx.ravel() + 1j * yy.ravel()

    # Generate grid points for each zone
    Qus = np.zeros([len(zone_centers)] + list(base_qs.shape), dtype=complex)
    for u, zone_center in enumerate(zone_centers):
        Qus[u] = zone_center + base_qs

    return Qus

def generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax, Ns, num_samples=2000):
    """
    Generate covariance matrices for all possible multiplicities.

    Parameters:
    side : float
        Side length of zones.
    nus : np.ndarray
        AP positions.
    zone_centers : np.ndarray
        Zone centers.
    A : int
        Number of antennas per AP.
    F : int
        Total number of antennas.
    rho : float
        Path loss exponent.
    d0 : float
        Reference distance.
    Kmax : int
        Maximum multiplicity.
    Ns : int
        Number of samples per k.
    num_samples : int, optional
        Number of grid points (default: 2000).

    Returns:
    tuple
        positions_for_ks : list of np.ndarray
            Sampled sensor positions for each multiplicity k.
        all_Covs : np.ndarray
            Covariance matrices for all zones and multiplicities.
    """
    U = len(zone_centers)
    Qus = generate_uniform_grid(side, num_samples, zone_centers)
    positions_for_ks = [np.array([np.vstack([np.random.choice(Qus[u], size=(k)) for _ in range(Ns)]) for u in range(U)]) for k in range(1,Kmax+1)]
    all_Covs = np.array([generate_cov_matrix(positions_for_ks[k], nus, A, rho=rho, d0=d0).sum(axis=-2) for k in range(Kmax)])
    all_Covs = np.vstack((np.zeros((1, U, Ns, F)), all_Covs))
    all_Covs = np.vstack([np.expand_dims(all_Covs[:,u,:,:],axis=0) for u in range(U)])
    return positions_for_ks, all_Covs
