# Geochemical Analysis and miniRUEDI Processing

This repository provides Python notebooks and utility functions for analyzing seasonal geochemical dynamics in deep aquifer systems, focusing on gas compositions and conductivity measurements from high-frequency miniRUEDI data.

---

## Repository Structure

```
.
├── src/
│   ├── config.py         # Project-level configuration (paths, constants)
│   └── functions.py      # Shared utility functions for data processing and analysis
├── geochemical_analysis.ipynb      # Main notebook for gas-conductivity correlation analysis
├── miniRUEDI_analysis.ipynb        # Data cleaning, normalization, and resampling of miniRUEDI datasets
├── .gitignore            # Files and folders excluded from version control
└── README.md             # This documentation file
```

---

## Overview

This project consists of two main workflows:

- **`geochemical_analysis.ipynb`**  
    Analyzes seasonal behavior of normalized gas ratios (e.g., CO₂/N₂, CH₄/N₂) alongside electrical conductivity measurements. Includes correlation matrix visualizations and seasonal sinusoidal model fitting.

- **`miniRUEDI_analysis.ipynb`**  
    Handles loading, processing, and filtering of raw miniRUEDI gas measurement data. Resamples to uniform time intervals and prepares cleaned datasets for further geochemical analysis.

---

## Setup

1. **Clone the repository:**
     ```bash
     git clone https://github.com/your-username/your-repo-name.git
     cd your-repo-name
     ```

2. **Create and activate a virtual environment (recommended):**
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows: venv\Scripts\activate
     ```

3. **Install dependencies:**
     ```bash
     pip install -r requirements.txt
     ```

4. *(Optional)* Create a `.env` file with custom configuration if required by `config.py`.

---

## Example Analyses

- Plot normalized gas time series with seasonal fits
- Generate correlation matrices between gas species
- Link miniRUEDI gas readings with conductivity changes across seasons

---

## Dependencies

Core libraries include:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scipy`, `sklearn`
- `matplotlib.dates`, `IPython.display`

All required packages are listed in `requirements.txt`.

---

## Development Notes

The `src/functions.py` file contains reusable components such as:

- Seasonal model fitting (`fitter`, `sinusoidal_func`)
- Linear regression overlays
- Resampling and filtering routines
- Data preprocessing for miniRUEDI output

If you extend this project, consider modularizing new functionality here.

---

## Author

Sébastien Giroud — ETH Zürich / Eawag

If you use this code in a publication, please cite accordingly.
