"""
Test Script for Distributed Decoder

This script demonstrates the usage of the distributed decoder
in a single simulation scenario. Modify parameters to test other cases.

Author: Kaan Okumus
Date: April 2025
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TUMA_simulation import TUMA_simulator

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
nAMPIter = 10  # Number of AMP iterations
Ns = 500  # Number of Monte Carlo samples for covariance approximation
nMCs = 100  # Number of Monte Carlo runs
Kmax = 5  # Maximum multiplicity with non-neglible prior, you can check looking at priors

# Simulate centralized decoder
print("Running Distributed Decoder Simulation...")
decoder_type = "distributed" # Distributed decoder
tv_dists = TUMA_simulator(
        Ju=Ju, SNR_rx_dB=SNR_rx_dB, Mau=Mau, Kau=Kau, A=A, n=n, side=side, rho=rho, d0=d0, 
        P=P, nAMPIter=nAMPIter, Ns=Ns, nMCs=nMCs, Kmax=Kmax, 
        decoder_type=decoder_type,
        display_topology=False, # Set to True to visualize topology
        withOnsager=True, # Set to True to include Onsager term
        plot_perf=True # Set to True to visualize per AMP iteration performance and true and estimated types in the same plot at the end
)

# Print results
print("\nSimulation Complete!")
print(f"TV Distances: {tv_dists}")
print(f"Average TV Distance): {np.mean(tv_dists):.4f}")