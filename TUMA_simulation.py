"""
TUMA Simulation Script

This script simulates the TUMA framework over a fading cell-free massive MIMO setup.
It supports both centralized and distributed decoders using AMP-based Bayesian estimation.

Author: Kaan Okumus
Date: April 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.helper_cf_tuma_tx import transmit, extract_user_positions
from utils.helper_topology import setup_topology, plot_topology
from utils.helper_decoder import generate_all_covs, generate_priors

from centralized_decoder import centralized_decoder
from distributed_decoder import distributed_decoder

def TUMA_simulator(
        Ju, SNR_rx_dB, Mau, Kau, A, n, side, rho, d0, P, nAMPIter, Ns, nMCs, 
        decoder_type,
        display_topology=False, force_Kmax=3, Kmax=3, Ns_smaller=1, withOnsager=False, plot_perf=False, print_perf=False
):
    """
    Simulate TUMA system with fading and message collisions.

    Parameters:
    - Ju : int
        Log2 of the number of quantized positions per zone.
    - SNR_rx_dB : float
        Target receive SNR in dB.
    - Mau, Kau : int
        Number of users and total transmitted messages per zone.
    - A : int
        Number of antennas per AP.
    - n : int
        Blocklength (number of channel uses).
    - side : float
        Length of the coverage area.
    - rho, d0 : float
        Pathloss parameters.
    - P : float
        Transmit power per user.
    - nAMPIter : int
        Number of AMP iterations.
    - Ns : int
        Number of samples for covariance estimation.
    - nMCs : int
        Number of Monte Carlo simulations.
    - decoder_type : str
        "centralized" or "distributed" decoder selection.
    - display_topology : bool, optional
        If True, displays the topology for the first Monte Carlo run.
    - force_Kmax, Kmax : int, optional
        Maximum multiplicity constraints.
    - Ns_smaller : int, optional
        Number of samples for residual covariance approximation.
    - withOnsager : bool, optional
        Whether to include Onsager correction in AMP.
    - plot_perf : bool, optional
        Whether to plot performance metrics during decoding.
    - print_perf : bool, optional
        Whether to print performance values during decoding.

    Returns:
    - tv_dists : np.ndarray
        Array of Total Variation distances across Monte Carlo runs.
    """
    tv_dists = np.zeros(nMCs)
    for idxMC in range(nMCs):
        print(f"Monte Carlo Run {idxMC + 1}/{nMCs}")

        # Setup topology
        U, B, zone_centers, nus = setup_topology(
            side=side, 
            topology_type=2, 
            mult=2, jitter=0.0, multiple_zone=True,
            rows=3, cols=3
        )

        # System parameters
        F = B * A
        Mus = [2**Ju] * U
        Maus = [Mau]*U
        Kaus = [Kau]*U
        Mu = 2**Ju
        Mus = [Mu]*U   

        # SNR, power parameters
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
        Y, k_true, X_true, zones_infos = transmit(
            Mus, Maus, Kaus, side, n, F, A, U, Cx, nP, sigma_w, nus, zone_centers, rho, d0, margin=side/10, force_Kmax=force_Kmax
        )

        # Display topology for the first Monte Carlo run if enabled
        if idxMC == 0 and display_topology:
            user_positions = extract_user_positions(zones_infos)
            plot_topology(nus, side, zone_centers, user_positions, 2, 3, 3)
        
        # Covariance approximation
        _, all_Covs = generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax=Kmax, Ns=Ns)
        _, all_Covs_smaller = generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax=Kmax, Ns=Ns_smaller)

        # Prior computation
        priors = generate_priors(
            Kaus, Maus, Mus, Kmax=Kmax
        )
        log_priors = [np.log(priors_u) for priors_u in priors]

        # Decoder
        if decoder_type=="centralized":
            est_k = centralized_decoder(
                Y=Y, Mus=Mus, nAMPIter=nAMPIter, B=B, A=A, Cx=Cx, Cy=Cy, nP=nP, 
                priors=priors, log_priors=log_priors, all_Covs=all_Covs, all_Covs_smaller=all_Covs_smaller, 
                withOnsager=withOnsager, k_true=k_true, X_true=X_true, sigma_w=sigma_w, plot_perf=plot_perf, print_perf=print_perf)
            print("Centralized decoder")
        elif decoder_type=="distributed":
            est_k = distributed_decoder(
                Y=Y, M=Mu, U=U, nAMPIter=nAMPIter, B=B, A=A, Cx=Cx, Cy=Cy, nP=nP, 
                priors=priors, log_priors=log_priors, all_Covs=all_Covs, all_Covs_smaller=all_Covs_smaller, 
                withOnsager=withOnsager, X_true=X_true, sigma_w=sigma_w, plot_perf=plot_perf, print_perf=print_perf
            )
            print("Distributed decoder")
        
        # Compute TV distance
        def obtain_global_type(k, U):
            global_k = k.reshape(U,-1).sum(axis=0)
            global_k = global_k/(global_k.sum())
            return global_k
        tv_dists[idxMC] = np.abs(obtain_global_type(k_true,U) - est_k/(est_k.sum())).sum()/2

        if plot_perf:
            plt.figure()
            plt.stem(obtain_global_type(k_true,U), label="True")
            plt.stem(est_k/(est_k.sum()), "r--", label="Est")
            plt.legend()
            plt.grid("True")
            plt.title(f"True and estimated types for {decoder_type} decoder")
            plt.xlabel("Message indices")
            plt.ylabel("Type")
            plt.show()

        print(f"\tTv distance = {tv_dists[idxMC]}")
    return tv_dists
