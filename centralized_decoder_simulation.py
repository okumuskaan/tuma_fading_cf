"""
Centralized Decoder Simulation

This script implements and simulates the centralized decoder for the TUMA framework with a fading 
cell-free massive MIMO setup. It uses AMP iterations with Bayesian estimation.

Author:
Kaan Okumus
Date:
January 2025
"""

import numpy as np
import scipy.stats

from utils.helper_bayesian_denoiser import *
from utils.helper_cf_tuma_tx import *
from utils.helper_topology import *

def generate_cov_matrix(pos, nus, A, rho=3.67, d0=0.01357):
    """
    Generate covariance matrix based on user positions.

    Parameters:
    pos : np.ndarray
        Array of user positions in complex form (real + j*imag).
    nus : np.ndarray
        Array of antenna positions in complex form.
    A : int
        Number of antennas per access point.
    rho : float, optional
        Path loss exponent (default: 3.67).
    d0 : float, optional
        Reference distance for path loss (default: 0.01357).

    Returns:
    np.ndarray
        Covariance matrix with dimensions based on input positions.
    """
    pos_shape = list(pos.shape)
    return (gamma(np.expand_dims(pos,axis=(-1,-2)), nus.reshape(-1,1), rho=rho, d0=d0)*np.ones((1,A))).reshape(pos_shape + [-1])

def generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax, Ns, num_samples=2000):
    """
    Generate covariance matrices for all zones using sampling-based approximation.

    Parameters:
    side : float
        Length of the side of the simulation area.
    nus : np.ndarray
        Array of antenna positions.
    zone_centers : np.ndarray
        Array of zone center positions.
    A : int
        Number of antennas per access point.
    F : int
        Total number of antennas (zones * antennas per zone).
    rho : float
        Path loss exponent.
    d0 : float
        Reference distance for path loss.
    Kmax : int
        Maximum multiplicity (number of sources per message).
    Ns : int
        Number of samples for covariance estimation.
    num_samples : int, optional
        Number of uniform grid points (default: 2000).

    Returns:
    tuple
        - positions_for_ks : List of sampled positions for each \( k \).
        - all_Covs : Covariance matrices for all possible multiplicities.
    """
    U = len(zone_centers)
    Qus = generate_uniform_grid(side, num_samples, zone_centers)
    positions_for_ks = [np.array([np.vstack([np.random.choice(Qus[u], size=(k)) for _ in range(Ns)]) for u in range(U)]) for k in range(1,Kmax+1)]
    all_Covs = np.array([generate_cov_matrix(positions_for_ks[k], nus, A, rho=rho, d0=d0).sum(axis=-2) for k in range(Kmax)])
    all_Covs = np.vstack((np.zeros((1, U, Ns, F)), all_Covs))
    all_Covs = np.vstack([np.expand_dims(all_Covs[:,u,:,:],axis=0) for u in range(U)])
    return positions_for_ks, all_Covs


def compute_prior(Ka, Ma, M, uniform=False, Kmax=None):
    """
    Compute prior for message multiplicities.

    Parameters:
    Ka : int
        Total number of transmitted messages.
    Ma : int
        Number of users per zone.
    M : int
        Total number of quantized message positions (e.g., grid size).
    uniform : bool, optional
        If True, assigns a uniform prior for multiplicities (default: False).
    Kmax : int, optional
        Maximum multiplicity (default: None).

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
    Kaus : list[int]
        List of total transmitted messages for each zone.
    Maus : list[int]
        List of users per zone.
    Mus : list[int]
        List of quantized message positions for each zone.
    Kmax : int
        Maximum multiplicity (number of sources per message).

    Returns:
    list[np.ndarray]
        List of prior distributions for each zone.
    """
    U = len(Kaus)
    priors = [
        compute_prior(Kaus[u], Maus[u], Mus[u])[: Kmax + 1].reshape(-1) for u in range(U)
    ]
    priors = [prior / prior.sum() for prior in priors]
    priors = [prior.reshape(1, -1) * np.ones((Mus[0], 1)) for prior in priors]
    return priors


def generate_uniform_grid(side, num_points, zone_centers, margin=0):
    """
    Generate a uniform grid of points for all zones.

    Parameters:
    side : float
        Length of the side of the simulation area.
    num_points : int
        Total number of grid points to generate across the area.
    zone_centers : np.ndarray
        Array of complex numbers representing the centers of the zones.
    margin : float, optional
        Margin to apply within the grid to avoid edge effects (default: 0).

    Returns:
    np.ndarray
        An array of grid points, where the first dimension corresponds to zones,
        and the remaining dimensions represent the grid points within each zone.
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


# Residual covariance
def compute_T(Z, n, A, B):
    c = np.diag(Z.conj().T @ Z).real/n
    taus = np.mean(c.reshape(B,A),axis=1)
    taus = (np.ones((1,A))*(taus.reshape(-1,1))).reshape(-1)
    return taus


# Centralized Decoder Function
def centralized_decoder(Y, Mus, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors, all_Covs, withOnsager=False, k_true=None):
    """
    Centralized Decoder with Multisource AMP and Bayesian Estimation.

    Parameters:
    Y : np.ndarray
        Received signal matrix.
    Mus : list[int]
        Number of codewords per zone.
    nAMPIter : int
        Number of AMP iterations.
    B : int
        Number of access points (APs).
    A : int
        Number of antennas per AP.
    Cx, Cy : functions
        Encoding and decoding functions.
    nP : float
        Normalized power per transmission.
    priors : list[np.ndarray]
        Priors for each zone.
    all_Covs : np.ndarray
        Covariance matrices for sampling-based approximation.
    withOnsager:
        Boolean variable for including Onsager term
    k_true:
        True multiplicity vector.

    Returns:
    est_k : np.ndarray
        Estimated multiplicity vector.
    """
    # Initialization
    n, F = Y.shape
    U = len(Mus)
    Z = Y.copy()  # Residual signal
    est_X = [np.zeros((Mus[u], F), dtype=complex) for u in range(U)]
    tv_dists = []

    for t in range(nAMPIter):
        print(f"AMP Iteration {t + 1}/{nAMPIter}")
        
        # Residual Covariance
        T = compute_T(Z, n, A, B)

        # AMP Updates
        Gamma = np.zeros_like(Z)
        R = []
        for u in range(U):
            R_u = Cy(Z, u) + np.sqrt(nP) * est_X[u]
            R.append(R_u)
            Qu = np.zeros((F,F), dtype=complex)
            for m in range(Mus[u]):
                if withOnsager:
                    est_X[u][m], est_k, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
                        R_u[m], all_Covs[u], T, nP, log_priors[u][m]
                    )
                    Qu += Qum
                else:
                    est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(R_u[m], all_Covs[u], T, nP, log_priors[u][m])

            Gamma += Cx(est_X[u], u)
            Gamma -= (1/n) * Z @ Qu

        Z = Y - np.sqrt(nP) * Gamma

        # Estimate multiplicity/type
        est_k, posteriors = estimate_type_samplingbased_logsumexp(R, T, all_Covs, Mus, nP, log_priors)

        # Update priors
        damp=0.9
        priors = [damp*priors[u] + (1-damp)*posteriors[u] for u in range(U)]
        log_priors = [np.log(priors[u]) for u in range(U)]

        # TV Distance
        if k_true is not None:
            tv_dist = np.sum(np.abs(k_true / np.sum(k_true) - est_k / np.sum(est_k))) / 2
            tv_dists.append(tv_dist)

        # Convergence Check
        if len(tv_dists) > 2 and all(np.isclose(tv_dists[-1], d) for d in tv_dists[-3:]):
            break

    return est_k


# Simulation Function
def simulate_centralized_decoder(
        Ju, SNR_rx_dB, Mau, Kau, A, n, side, rho, d0, P, nAMPIter, Ns, nMCs, 
        display_topology, force_Kmax, Kmax, 
        topology_type=2, rows=3, cols=3, withOnsager=False, mult=2, jitter=0.0, multiple_zone=True
    ):
    """
    Simulates the centralized decoder over multiple Monte Carlo runs.

    Parameters:
    Ju, SNR_rx_dB, Mau, Kau, A, n, side, rho, d0, P : float
        Various system parameters (see centralized_decoder for details).
    nAMPIter : int
        Number of AMP iterations.
    Ns : int
        Number of samples for covariance approximation.
    nMCs : int
        Number of Monte Carlo runs.
    topology_type, rows, cols, jitter, mult, multiple_zone:
        Topology parameters.

    Returns:
    tv_dists : np.ndarray
        Total variation distances for each Monte Carlo run.
    """
    tv_dists = np.zeros(nMCs)

    for idxMC in range(nMCs):
        print(f"Monte Carlo Run {idxMC + 1}/{nMCs}")

        # Setup Topology
        U, B, zone_centers, nus = setup_topology(
            side=side, 
            topology_type=topology_type, 
            mult=mult, jitter=jitter, multiple_zone=multiple_zone,
            rows=rows, cols=cols
        )

        F = B * A
        Mus = [2**Ju] * U
        Maus = [Mau]*U
        Kaus = [Kau]*U
        Mu = 2**Ju
        Mus = [Mu]*U        

        SNR_rx = 10**(SNR_rx_dB/10)
        min_dist = np.abs(np.array([(1+1j)*0.0]) - nus).min()
        SNR_tx = SNR_rx * (1 + (min_dist/d0)**rho)
        sigma_w = np.sqrt(P/SNR_tx)
        nP = n * P

        # Prepare the codebook
        M = sum(Mus)
        C = np.random.randn(n,M) + 1j * np.random.randn(n,M)
        C = C/np.linalg.norm(C,axis=0)
        def Cx(x,u=0):
            return C[:,u*Mus[u]:(u+1)*Mus[u]]@x
        def Cy(y,u=0):
            return (C[:,u*Mus[u]:(u+1)*Mus[u]].conj().T)@y

        # Transmission
        margin = side/10 if topology_type!=3 else -side/20
        Y, k_true, _, zones_infos = transmit(
            Mus, Maus, Kaus, side, n, F, A, U, Cx, nP, sigma_w, nus, zone_centers, rho, d0, margin=margin, force_Kmax=force_Kmax
        )

        # Display topology for the first Monte Carlo run if enabled
        if idxMC == 0 and display_topology:
            user_positions = extract_user_positions(zones_infos)
            plot_topology(nus, side, zone_centers, user_positions, topology_type, rows, cols)

        # Covariance Approximation
        _, all_Covs = generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax=Kmax, Ns=Ns)

        # Decoder
        priors = generate_priors(
            Kaus, Maus, Mus, Kmax=Kmax
        )
        log_priors = [np.log(priors_u) for priors_u in priors]
        est_k = centralized_decoder(Y, Mus, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors, all_Covs, withOnsager=withOnsager, k_true=k_true)

        # Compute TV distance
        def obtain_global_type(k, U):
            global_k = k.reshape(U,-1).sum(axis=0)
            global_k = global_k/(global_k.sum())
            return global_k
        
        tv_dists[idxMC] = np.sum(np.abs(obtain_global_type(est_k, U) - obtain_global_type(k_true, U))) / 2



        print(f"TV distance: {tv_dists[idxMC]:.4f}")

    return tv_dists