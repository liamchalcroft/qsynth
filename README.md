<div align="center">

# 🧠 Domain-Agnostic Stroke Lesion Segmentation
## Using Physics-Constrained Synthetic Data

[![Paper](https://img.shields.io/badge/MICCAI-2025-blue.svg)](https://miccai2025.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*A framework for robust stroke lesion segmentation across heterogeneous MRI domains using physics-based synthetic data generation*

[Overview](#overview) | [Installation](#installation) | [Usage](#usage) | [Results](#results)

</div>

---

> **Note:** This is a preliminary release of our research code. Detailed model descriptions, trained weights, and comprehensive documentation will be made available upon publication of our paper at MICCAI 2025.

## Overview

We present two novel methods for domain-agnostic stroke lesion segmentation:
1. **qATLAS**: A neural network that estimates qMRI maps from standard MPRAGE images
2. **qSynth**: A direct synthesis approach for qMRI maps using label-conditioned Gaussian mixture models

Both methods leverage physics-based forward models to ensure physical plausibility in the simulated images.

<details>
<summary><b>🔍 Key Features</b></summary>

- Physics-constrained synthetic data generation
- Robust performance across multiple MRI modalities
- Domain-agnostic segmentation capabilities
- Extensive validation on clinical datasets
</details>
