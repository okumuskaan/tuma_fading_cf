# Type-Based Unsourced Multiple Access Over Fading Channels with Cell-Free Massive MIMO

## Overview

This repository contains the implementation of the algorithm presented in our paper, [**Type-Based Unsourced Multiple Access (TUMA) Over Fading Channels with Cell-Free (CF) Massive MIMO**](https://ieeexplore.ieee.org/document/11195493), presented at ISIT 2025.

The project includes simulations and decoding algorithms tailored for large-scale random access communication under fading channels using centralized and distributed decoders. The proposed decoders builds upon multisource approximate message passing (AMP), which is adapted here to handle message collisions — a critical challenge in unsourced random access. Our decoder uses Bayesian estimation to recover both the message list and their transmission multiplicities in CF MIMO systems.

## Key Features

- **Cell-Free Massive MIMO support:** Includes topology and channel modeling with Rayleigh fading.
- **Centralized and Distributed Decoders:** Based on multi-source AMP with Bayesian denoisers.
- **Message Collisions:** Explicitly modeled and recovered through a type-based approach.
- **Monte Carlo Simulations:** For empirical performance evaluation over realistic setups.

## Project Structure

```
.
├── utils/
│   ├── helper_bayesian_denoiser.py         # Bayesian denoising functions with Onsager term
│   ├── helper_cf_tuma_tx.py                # Transmitter-side signal and message generation
│   ├── helper_topology.py                  # Topology creation and visualization tools
│   └── helper_decoder.py                   # Covariance, priors, and helper functions for decoding
│
├── centralized_decoder.py                  # Centralized AMP-based decoder for TUMA
├── distributed_decoder.py                  # Distributed AMP-based decoder for TUMA
├── TUMA_simulation.py                      # Simulation script for TUMA centralized and distributed decoders
├── amp_da_simulation.py                    # AMP-DA decoding and simulation framework
│
├── tests/
│   ├── test_amp_da.py                      # AMP-DA decoding for comparison
│   ├── test_tuma_centralized.py            # Test centralized AMP decoder in TUMA
│   └── test_tuma_distributed.py            # Test disributed AMP decoder in TUMA
│
└── README.md                               # Project overview and usage instructions
```

## Usage

### Running Tests

Navigate to the project directory and run the test scripts from the ``tests/`` directory. For example,

```bash
python tests/test_tuma_centralized.py
```

### Simulations


* **Centralized Decoder Simulation:** Run ``tests/test_tuma_centralized.py`` to test the centralized decoder for a single scenario.

* **Distributed Decoder Simulation:** Run ``tests/test_tuma_distributed.py`` to test the distributed decoder.

* **AMP-DA Simulation:** Run ``tests/test_amp_da.py`` to test the AMP-DA decoder.


### Customization
Users can modify the test scripts to experiment with different parameters, such as the number of users, antennas, SNR, and topology settings.




## Citation

If you use this code or refer to the system in your work, please cite our paper:

```
@INPROCEEDINGS{tuma_fading_2025,
  author={Okumus, Kaan and Ngo, Khac-Hoang and Durisi, Giuseppe and Ström, Erik G.},
  booktitle={2025 IEEE International Symposium on Information Theory (ISIT)}, 
  title={Type-Based Unsourced Multiple Access Over Fading Channels with Cell-Free Massive MIMO}, 
  year={2025},
  month={June},
  doi={10.1109/ISIT63088.2025.11195493}
}
```

## References

- **Multisource AMP Algorithm**:
  B. Cakmak, E. Gkiouzepi, M. Opper, and G. Caire, "Joint Message Detection and Channel Estimation for Unsourced Random Access in Cell-Free User-Centric Wireless Networks," *IEEE Transactions on Information Theory*, 2025. [https://ieeexplore.ieee.org/abstract/document/10884602](https://ieeexplore.ieee.org/abstract/document/10884602)

- **AMP-DA Algorithm**:
  L. Qiao, Z. Gao, M. B. Mashadi, and D. Gunduz "Digital Over-the-Air Aggregation for Federated Edge Learning," *IEEE Journal on Selected Areas in Communications*, 2024. [https://ieeexplore.ieee.org/document/10648926](https://ieeexplore.ieee.org/document/10648926)

- **Code Reference**:
  AMP-DA codes based on [liqiao19/MD-AirComp](https://github.com/liqiao19/MD-AirComp)
