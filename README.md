# Polar Spectral Operators for Learning on Angular Domains

This repository contains the reference implementation accompanying the paper

> **Foundations of Polar Linear Algebra**<br>
> *A Structured Framework for Radial–Angular Operators*


The code provides a minimal implementation of polar and spectral operators,
together with reproducible experiments on MNIST illustrating different modeling
choices discussed in the paper.

All theoretical details and derivations are provided in the manuscript.  
This repository focuses on reproducibility and clarity.

---

## Repository Structure

The codebase is organized in two main parts:

### A) Minimal Operator Definitions

```text
polar/
├── operators.py   # rotors, polar dense / convolutional operators
├── spectral.py    # Fourier-domain utilities, gating, projections
└── tensor.py      # PolarTensor abstraction
```

This directory contains a lightweight, self-contained implementation of the
polar and spectral operators introduced in the paper.

### B) Experiments

```text
experiments/
├── mnist_polar/       # MNIST with polar operators (spatial + spectral)
├── mnist_spectral/    # Fully spectral MNIST pipeline
└── mnist_selfadjoint/ # Spectral MNIST with self-adjoint (real spectrum) kernels
```

Each experiment is implemented as an independent, reproducible training script.
MNIST is used purely as a controlled, low-dimensional testbed to isolate the
behavior of the proposed operators, not as a benchmark for state-of-the-art
performance.

---

## Environment Setup
We recommend using a Conda environment.

```bash
conda create -n polar python=3.10
conda activate polar
pip install -r requirements.txt
```
---

### Required packages include:
* PyTorch
* torchvision
* numpy
* scipy
* matplotlib (optional, for plots)

MNIST is automatically downloaded via torchvision.

---
### Quick Sanity Check

To verify that the environment is correctly set up, you can run a short sanity
training (few epochs, reduced dataset):
```bash
python experiments/mnist_spectral/train.py --epochs 2
```
---

## Running the Experiments
1) MNIST with Polar Operators

Polar operators acting on angular coordinates, with spectral diagonalization
used internally.
```bash
python experiments/mnist_polar/train.py
```
2) Fully Spectral MNIST Pipeline

The entire model operates in the Fourier domain after the initial transform.
```bash
python experiments/mnist_spectral/train.py
```
3) Self-Adjoint Spectral Model (Real Eigenvalues)

Spectral kernels are constrained to have real eigenvalues, enforcing
self-adjointness of the underlying operators.
```bash
python experiments/mnist_selfadjoint/train.py
```
This experiment demonstrates that real-valued spectral operators are sufficient
to reach competitive accuracy while reducing parameter count and computational
complexity.

---
### Outputs

By default, each run creates an output directory containing:
* training logs
* validation metrics
* best model checkpoint

Checkpoints are saved automatically when validation accuracy improves.

---
### Reproducibility

Fixed random seeds are used by default.
All experiments are deterministic up to GPU nondeterminism.
Hyperparameters are defined directly in the experiment scripts.

---
## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{polar_spectral_operators,
  title   = {Polar Algebra and Fully Spectral Neural Models},
  author  = {Giovanni Guasti},
  journal = {Journal / arXiv},
  year    = {2025}
}
```
A machine-readable citation is provided in `CITATION.cff`.

---
## License

This code is released under an MIT license.



