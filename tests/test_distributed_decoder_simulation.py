"""
Test Script for Distributed Decoder

This script demonstrates the usage of the distributed decoder and simulations for a single scenario.
Users can modify the parameters to test other scenarios.
"""

import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distributed_decoder_simulation import simulate_distributed_decoder
import numpy as np

# Example parameters
Ju = 8
SNR_rx_dB = 10.0
Mau = 13
Kau = 20
A = 4
n = 1024
side = 0.1
rho = 3.67
d0 = 0.01357
P = 1/n
nAMPIter = 5
Ns = 500
topology_type = 2
multiple_zone = True
mult = 2
jitter = 0.0
rows=3
cols=3
transceiver_type = 2
display_topology = True
plot_performance = False
display_type_estimation = False
nMCs = 10
Kmax = 5
force_Kmax = 3

# Simulate distributed decoder
print("Running Distributed Decoder Simulation...")
tv_dists = simulate_distributed_decoder(
    side=side, SNR_rx_dB=SNR_rx_dB, Ju=Ju, Mau=Mau, Kau=Kau, nAMPIter=nAMPIter, 
    A=A, n=n, P=P, nMCs=nMCs, d0=d0, rho=rho, force_Kmax=3, Kmax=5, Ns=Ns, display_topology=display_topology
)

# Output results
print("\nSimulation Complete!")
print(f"TV Distances: {tv_dists}")
print(f"Average TV Distance): {np.mean(tv_dists):.4f}")
