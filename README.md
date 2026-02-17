# Stellar Memory of Galactic Spiral Shocks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for the paper **"Stellar Memory of Galactic Spiral Shocks: A Gaia DR3 Analysis of Classical Cepheids"** by Chi et al. (2025). We present a high‑precision kinematic analysis of Classical Cepheids using Gaia DR3 data to test whether young stars retain kinematic imprints of spiral‑induced gas shocks.

## Project Overview

The dynamic nature of the Milky Way’s spiral arms—whether they are quasi‑stationary density waves or transient material features—remains debated. Gas dynamics react non‑linearly to spiral potentials, forming shocks, but similar signatures in stellar populations have been difficult to detect due to their higher velocity dispersion. In this work, we introduce a **Local Phase‑Space Reconstruction (LPSR)** technique that uses the dynamically relaxed Red Clump (RC) population as a local velocity background. By subtracting this background from the kinematics of Classical Cepheids (young, bright tracers of spiral structure), we isolate the differential motion associated with the spiral perturbation.

Our main findings:
- A statistically significant radial velocity residual ($\Delta V_R$) amplitude of $\sim 10\,\mathrm{km\,s^{-1}}$ is detected.
- Cepheids show a distinct inflow ($\Delta V_R < 0$) in the pre‑arm region ($-\pi/2 < \theta_{\rm sp} < 0$) and a sharp reversal near the arm center.
- This “kinematic decoupling” between young and old stars supports the density‑wave scenario with strong shocks, indicating that young stellar populations inherit the hydrodynamic shock signature of their natal gas.

## Dependencies

The code is written in **Python 3.9+** and requires the following packages:

- `numpy`, `scipy` – numerical computations
- `astropy` – astronomical coordinate transformations and units
- `gala` – Galactic dynamics (potential models, orbit integration)
- `pandas` – data manipulation
- `matplotlib`, `seaborn` – plotting
- `astroquery` – optional, for querying Gaia data
- `tqdm` – progress bars (optional)

All dependencies can be installed via `pip`:

```bash
pip install numpy scipy astropy gala pandas matplotlib seaborn astroquery tqdm
License

This project is licensed under the MIT License – see the LICENSE file for details.

Citation

If you use this code or data in your research, please cite the following paper:

Chi, H., Wang, F., et al. 2025, Stellar Memory of Galactic Spiral Shocks: A Gaia DR3 Analysis of Classical Cepheids, A&A Letters, in press.

BibTeX entry will be provided upon publication.

Contact

For questions or issues, please open an issue on GitHub or contact Huanbin Chi at chihuanbin@126.com.

---

## 📄 LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 Huanbin Chi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
