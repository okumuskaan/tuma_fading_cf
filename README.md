# TUMA with Fading in Cell-Free Systems

## Overview
This repository contains the code implementation for the Type-Based Unsourced Multiple Access (TUMA) framework with fading channels in a cell-free (CF) massive MIMO system. The repository provides algorithms and simulations for centralized and distributed decoders using Approximate Message Passing (AMP) and Bayesian estimation.


## Project Structure

The project is organized as follows:

### Root Directory
- **README.md**: This file provides an overview of the project, its structure, and usage.

### `utils/` Folder
Contains helper files used across different modules:
- **`helper_bayesian_denoiser.py`**: Implements Bayesian denoiser functions, including sampling-based and Onsager correction approximations.
- **`helper_cf_tuma_tx.py`**: Provides utility functions for the transmitter, including generating sensor positions, multiplicities, and transmitted signals.
- **`helper_topology.py`**: Contains functions for creating and visualizing topologies, including hexagonal grids and user placements.
- **`amp_da_simulation.py`**: Implements the AMP-DA decoder and simulation framework.
- **`centralized_decoder_simulation.py`**: Implements centralized decoding for TUMA with Monte Carlo simulations.
- **`distributed_decoder_simulation.py`**: Implements distributed decoding inspired by Erik Larsson’s distributed AMP for TUMA.

### `tests/` Folder
Contains test scripts to verify and demonstrate the functionality of the different components:
- **`test_amp_da.py`**: Example script for running the AMP-DA algorithm for a single scenario.
- **`test_centralized_decoder_simulation.py`**: Example script for running the centralized decoder simulations.
- **`test_distributed_decoder_simulation.py`**: Example script for running distributed decoder simulations.


## Setup

### Requirements
- Python 3.8 or later
- Required Python libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`


## Usage

### Running Tests

Navigate to the project directory and run the test scripts from the ``tests/`` directory. For example,

```bash
python tests/test_centralized_decoder_simulation.py
```

### Simulations


* **Centralized Decoder Simulation:** Use ``test_centralized_decoder_simulation.py`` to evaluate the centralized decoder with TUMA.

* **Distributed Decoder Simulation:** Run ``test_distributed_decoder_simulation.py`` for distributed decoding scenarios.


* **AMP-DA Simulation:** Run ``test_amp_da.py`` to test the AMP-DA decoder for a specific scenario.

### Customization
Users can modify the test scripts to experiment with different parameters, such as the number of users, antennas, SNR, and topology settings.



## References

The AMP-DA algorithm was presented in:

- L. Qiao, J. Zhang, and K. B. Letaief, "Digital Over-the-Air Aggregation for Federated Edge Learning," *IEEE Transactions on Wireless Communications*, 2023. [DOI: 10.1109/TWC.2023.10648926](https://ieeexplore.ieee.org/document/10648926)

The implementation of AMP-DA in this repository is based on the code from the MD-AirComp project:

- GitHub Repository: [liqiao19/MD-AirComp](https://github.com/liqiao19/MD-AirComp)
