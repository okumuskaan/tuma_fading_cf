"""
Centralized Decoder Simulation

This script implements and simulates the centralized decoder for the TUMA framework 
in a fading cell-free massive MIMO setup. It applies AMP iterations with Bayesian estimation 
and supports Onsager correction.

Functions:
- compute_T: Estimate residual noise covariance matrix.
- centralized_decoder: Perform centralized decoding using multisource AMP and Bayesian updates.

Usage:
This module is intended for use in the TUMA framework and can be imported as:
    from centralized_decoder import centralized_decoder

Author: Kaan Okumus
Date: April 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.helper_bayesian_denoiser import *
from utils.helper_cf_tuma_tx import transmit
from utils.helper_topology import setup_topology, plot_topology
from utils.helper_decoder import generate_all_covs, generate_priors

def compute_T(Z, n, A, B):
    """
    Estimate the residual noise covariance.

    Parameters:
    Z : np.ndarray
        Residual matrix.
    n : int
        Blocklength.
    A : int
        Number of antennas per AP.
    B : int
        Number of access points.

    Returns:
    np.ndarray
        Estimated residual noise covariance vector.
    """
    c = np.diag(Z.conj().T @ Z).real/n
    taus = np.mean(c.reshape(B,A),axis=1)
    taus = (np.ones((1,A))*(taus.reshape(-1,1))).reshape(-1)
    return taus

def centralized_decoder(Y, Mus, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors, all_Covs, all_Covs_smaller, 
                        withOnsager=False, k_true=None, X_true=None, sigma_w=None, plot_perf=False, print_perf=False):
    """
    Centralized decoder using multisource AMP and Bayesian denoising.

    Parameters:
    Y : np.ndarray
        Received signal matrix.
    Mus : list of int
        Number of messages per zone.
    nAMPIter : int
        Number of AMP iterations.
    B : int
        Number of access points.
    A : int
        Number of antennas per AP.
    Cx, Cy : functions
        Encoding and decoding operators.
    nP : float
        Total transmission power.
    priors : list of np.ndarray
        Prior distributions for multiplicities.
    log_priors : list of np.ndarray
        Logarithm of prior distributions.
    all_Covs : np.ndarray
        Covariance matrices for posterior estimation.
    all_Covs_smaller : np.ndarray
        Covariance matrices for channel estimation.
    withOnsager : bool, optional
        Whether to apply Onsager correction (default: False).
    k_true : np.ndarray, optional
        True multiplicity vector (for evaluation).
    X_true : list of np.ndarray, optional
        True effective fading channel matrices (for evaluation).
    sigma_w : float, optional
        Noise standard deviation (for evaluation).
    plot_perf : bool, optional
        Whether to plot performance metrics (default: False).
    print_perf : bool, optional
        Whether to print performance values (default: False).

    Returns:
    -------
    est_k : np.ndarray
        Estimated multiplicity vector after decoding.
    """
    # Initialization
    n, F = Y.shape
    P = nP/n
    U = len(Mus)
    Z = Y.copy()
    est_X = [np.zeros((Mus[u], F), dtype=complex) for u in range(U)]
    tv_dists = []
    channel_est_perfs = []
    channel_est_T_perfs = []

    if X_true is not None:
        channel_est_perfs.append(P * np.sum([np.linalg.norm(X_true[u] - est_X[u], 'fro')**2 for u in range(U)]))

    # AMP iterations
    for t in range(nAMPIter):
        print(f"\t\tAMP iteration: {t+1}/{nAMPIter}")

        # Residual covariance
        T = compute_T(Z, n, A, B)

        # AMP updates
        Gamma = np.zeros_like(Z)
        R = []
        for u in range(U):
            R_u = Cy(Z, u) + np.sqrt(nP) * est_X[u]
            R.append(R_u)
            Qu = np.zeros((F,F), dtype=complex)
            for m in range(Mus[u]):
                if withOnsager:
                    est_X[u][m], est_k, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
                        R_u[m], all_Covs_smaller[u], T, nP, log_priors[u][m]
                    )
                    Qu += Qum
                else:
                    est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(R_u[m], all_Covs_smaller[u], T, nP, log_priors[u][m])
            Gamma += Cx(est_X[u], u)
            Gamma -= (1/n) * Z @ Qu

        Z = Y - np.sqrt(nP) * Gamma

        # Type estimation
        est_k, posteriors = estimate_type_samplingbased_logsumexp(R, T, all_Covs, Mus, nP, log_priors)

        # Prior update (no damping for now)
        damp=1.0
        priors = [damp*priors[u] + (1-damp)*posteriors[u] for u in range(U)]
        log_priors = [np.log(priors[u]) for u in range(U)]

        # Performance metrics
        if k_true is not None:
            tv_dist = np.sum(np.abs(k_true / np.sum(k_true) - est_k / np.sum(est_k))) / 2
            tv_dists.append(tv_dist)
        
        if X_true is not None:
            channel_est_perfs.append(P * np.sum([np.linalg.norm(X_true[u] - est_X[u], 'fro')**2 for u in range(U)]))
        if sigma_w is not None:
            channel_est_T_perfs.append(np.sum(np.abs(T - (sigma_w**2))))

    # Final performance update
    if sigma_w is not None:
        channel_est_T_perfs.append(np.sum(np.abs(compute_T(Z, n, A, B) - (sigma_w**2))))

    # Plot performance if requested
    if plot_perf:
        if print_perf:
            print("channel_est_perfs =", channel_est_perfs)
            print("channel_est_T_perfs =", channel_est_T_perfs)
        plt.figure()
        plt.semilogy(np.arange(1, len(channel_est_perfs)+1), channel_est_perfs, label="Channel Est")
        plt.semilogy(np.arange(1, len(channel_est_perfs)+1), channel_est_T_perfs, "--", label="Residual Cov Est")
        plt.legend()
        plt.grid("True")
        plt.xlabel("AMP iteration number")
        plt.ylabel("Multisource AMP performance scores")
        plt.show()

        plt.figure()
        plt.plot(tv_dists)
        plt.ylabel("TV distance")
        plt.xlabel("AMP iteration number")
        plt.grid("True")
        plt.show()

    # Final multiplicity vector
    M = int(sum(Mus)/U)
    est_k = est_k.reshape(-1,M).sum(axis=0)

    return est_k
