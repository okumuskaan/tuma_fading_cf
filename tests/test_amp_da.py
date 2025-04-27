"""
Test Script for AMP-DA

This script demonstrates the usage of the AMP-DA algorithm
in a single simulation scenario. Modify parameters to test other cases.

Author: Kaan Okumus
Date: April 2025
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from amp_da_simulation import simulate_AMP_DA

# Example parameters
side = 0.1  # Length of the side of the simulation area (km)
A = 4  # Antennas per access point
Mau = 13  # Number of distinct messages per zone
Kau = 20  # Total transmitted messages per zone, or number of active users per zone
Ju = 8  # Bits per zone
SNR_rx_dB = 10.0  # Received SNR in dB
rho = 3.67  # Path loss exponent
d0 = 0.01357  # Reference distance for path loss (km)
n = 1024  # Block length
P = 1 / n  # Transmission power
nMCs = 100  # Number of Monte Carlo runs
perfect_CSI = False  # Imperfect CSI
phase_max = np.pi / 6  # Maximum phase shift

# Simulate AMP-DA in our setup with imperfect CSI
print("Running AMP-DA simulation for one scenario...")
tv_dists = simulate_AMP_DA(
    Ju=Ju, SNR_rx_dB=SNR_rx_dB, Mau=Mau, Kau=Kau, A=A, n=n, 
    side=side, rho=rho, d0=d0, P=P, nMCs=nMCs, phase_max=phase_max, 
    display_topology=False, # Set to True to visualize topology
    plot_perf=True # Set to True to visualize true and estimated types in the same plot
)

# Print results
print("\nSimulation Complete!")
print(f"TV Distances: {tv_dists}")
print(f"Average TV Distance): {np.mean(tv_dists):.4f}")