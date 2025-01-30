"""
Test Script for Centralized Decoder

This script demonstrates the usage of the centralized decoder and simulations for a single scenario.
Users can modify the parameters to test other scenarios.
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from centralized_decoder_simulation import simulate_centralized_decoder
import numpy as np


# Example parameters
side = 0.1  # Length of the side of the simulation area
topology_type = 2  # Grid topology
rows, cols = 3, 3  # Grid dimensions
A = 4  # Antennas per radio unit (RU)
Mau = 13  # Number of users per zone
Kau = 20  # Total transmitted messages per zone
Ju = 4  # Bits per zone (\( \text{Ju} = \log_2(\text{Mu}) \))
SNR_rx_dB = 10.0  # Receiver SNR in dB
rho = 3.67  # Path loss exponent
d0 = 0.01357  # Reference distance for path loss
n = 1024  # Block length
P = 1 / n  # Transmission power
nAMPIter = 20  # Number of AMP iterations
Ns = 500  # Number of Monte Carlo samples
nMCs = 100  # Number of Monte Carlo runs
Kmax = 5  # Maximum multiplicity
force_Kmax = Kau  # Force maximum multiplicity

# Simulate centralized decoder
print("Running Centralized Decoder Simulation...")
tv_dists = simulate_centralized_decoder(Ju, SNR_rx_dB, Mau, Kau, A, n, side, rho, d0, P, nAMPIter, Ns, nMCs, force_Kmax=force_Kmax, Kmax=Kmax, display_topology=True, topology_type=2, rows=3, cols=3)

# Output results
print("\nSimulation Complete!")
print(f"TV Distances: {tv_dists}")
print(f"Average TV Distance): {np.mean(tv_dists):.4f}")
