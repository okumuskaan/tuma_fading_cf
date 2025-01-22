"""
Test Script for AMP-DA

This script demonstrates the usage of the AMP-DA algorithm and simulation for a single scenario.
Users can modify the parameters to test other scenarios.
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from amp_da_simulation import simulate_AMP_DA
import numpy as np

# Example parameters
side = 0.1  # Length of the side of the simulation area
topology_type = 2  # Grid topology
rows, cols = 3, 3  # Grid dimensions
A = 4  # Antennas per radio unit (RU)
Mau = 13  # Number of users per zone
Kau = 20  # Total transmitted messages per zone
Ju = 8  # Bits per zone (\( \text{Ju} = \log_2(\text{Mu}) \))
SNR_rx_dB = 10.0  # Receiver SNR in dB
rho = 3.67  # Path loss exponent
d0 = 0.01357  # Reference distance for path loss
n = 1024  # Block length
P = 1 / n  # Transmission power
nMCs = 100  # Number of Monte Carlo runs
perfect_CSI = False  # Use perfect CSI
imperfection_model = "phase"  # CSI imperfection model
sigma_noise_e = 0.1  # Noise standard deviation for CSI imperfection
phase_max = np.pi / 6  # Maximum phase error

# Simulate AMP-DA
print("Running AMP-DA simulation for one scenario...")
tv_dists = simulate_AMP_DA(
    side=side,
    topology_type=topology_type,
    mult=2,
    jitter=0.0,
    multiple_zone=True,
    rows=rows,
    cols=cols,
    A=A,
    Mau=Mau,
    Kau=Kau,
    Ju=Ju,
    SNR_rx_dB=SNR_rx_dB,
    rho=rho,
    d0=d0,
    P=P,
    n=n,
    nMCs=nMCs,
    perfect_CSI=perfect_CSI,
    imperfection_model=imperfection_model,
    sigma_noise_e=sigma_noise_e,
    phase_max=phase_max,
    display_topology=False,  # Set to True to visualize topology
)

# Output results
print(f"\nSimulation Complete!")
print(f"Total Variation Distances (TV): {tv_dists}")
print(f"Average TV Distance: {np.mean(tv_dists):.4f}")