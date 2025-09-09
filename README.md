# Near-Heisenberg-limited Parallel Amplitude Estimation

This repository provides the reference implementation and scripts from our paper [**“Near-Heisenberg-limited parallel amplitude estimation with logarithmic depth circuit”**](https://doi.org/10.48550/arXiv.2508.06121). 
For details of the algorithm, please refer to our paper.


## File Structure and Usage
```text
.
├── PAE_sample.ipynb                 # Jupyter notebook demonstrating PAE
├── calculate_pae_query-comp.py      # Script to compute query complexity
├── calculate_pae_QSP-error.py       # Script to evaluate QSP error (β) numerically
├── plot_query-comp.py               # Plot script for query complexity comparison (for Fig.2)
├── plot_query-comp_beta_effect.py   # Plot script analyzing the effect of β (for Fig.S2)
├── plot_QSP-error.py                # Plot script for visualizing β vs L (for Fig.S3)
├── plot_T-L.py                      # Plot script for T-L curve (for Fig.S4)
├── LICENSE.md                       # BSD 3-Clause Clear License (project-wide)
├── README.md                        # Project documentation (this file)
│
├── pae/                             # Core implementation of the PAE algorithm
│   ├── circuit.py                   # Circuit builder for PAE
│   ├── qsp.py                       # Quantum Signal Processing (QSP) utilities (licensed under the MIT License)
│   ├── estimate.py                  # Classical post-processing of robust phase estimation
│   └── utils.py                     # Utility functions
│
└── results/                             # Precomputed results and generated plots
    ├── results_QSP-error_worst.csv      # Table for worst-case QSP error
    ├── result_a0.1464_K9_n-trial100.csv # Query complexity results (a = sin^2(π/8))
    ├── result_a0.0000_K9_n-trial100.csv # Query complexity results (a = 0)
    ├── graph_query_comp.pdf             # Plot: Query complexity comparison (Fig.2)
    ├── graph_query_comp_beta.pdf        # Plot: Beta parameter effect (Fig.S2)
    ├── graph_QSP-error.pdf              # Plot: QSP error behavior (Fig.S3)
    └── graph_T-L.pdf                    # Plot: T-L (qubit count vs. depth) trade-off (Fig.S4)
```

### Usage

To reproduce the numerical results in the paper:

1. **Install required packages**:  
    See the environment instructions in `requirements.txt`.

2. **Run numerical calculations**:  
   - `calculate_pae_query-comp.py`: Generates query complexity data.
   - `calculate_pae_QSP-error.py`: Computes QSP error (β) tables.

3. **Generate plots**:  
   - Run the corresponding plot_*.py scripts to generate figures saved in results/.

The `PAE_sample.ipynb` notebook demonstrates how to use the PAE framework, from circuit construction to estimation.


## License
- Default: **BSD 3-Clause Clear** — see `LICENSE`.  
  *Note: This license does **not** grant patent rights.*
- Exception: `pae/qsp.py` only — **MIT** (full license text embedded at the top of the file).  
  This file is a modified version of code originally published at [angle-sequence](https://github.com/alibaba-edu/angle-sequence).

## Reference
- Kohei Oshio, Kaito Wada, Naoki Yamamoto. "Near-Heisenberg-limited parallel amplitude estimation with logarithmic depth circuit." arXiv preprint arXiv:2508.06121 (2025)