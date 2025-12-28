"""
polar.tensor
============

Torch implementations of circular shift operators and DFT utilities.

Conventions:
- Shift operator R_k acts as (R_k x)[n] = x[(n - k) mod N]
  which corresponds to np.roll(x, +k) in NumPy.

In PyTorch, we implement the action directly with torch.roll and optionally
build the dense permutation matrix for debugging / small N.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------
# Shift operators (action)
# ---------------------------------------------------------------------

def apply_shift_1d(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Apply circular shift along a dimension, consistent with (R_k x)[n] = x[(n-k) mod N].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any shape).
    k : int
        Shift amount.
    dim : int
        Dimension along which to apply the shift.

    Returns
    -------
    y : torch.Tensor
        Shifted tensor.
    """
    return torch.roll(x, shifts=k, dims=dim)


def inverse_shift_1d(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Apply inverse shift R_{-k}.
    """
    return torch.roll(x, shifts=-k, dims=dim)


# ---------------------------------------------------------------------
# Shift operators (matrix form) - mostly for debugging / theory checks
# ---------------------------------------------------------------------

def shift_matrix(N: int, k: int, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Dense permutation matrix representing R_k with convention (R_k x)[n] = x[(n-k) mod N].

    Returns a tensor of shape (N, N).
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    k = k % N

    # Build explicitly to avoid ambiguity
    R = torch.zeros((N, N), device=device, dtype=dtype)
    idx = torch.arange(N, device=device)
    R[idx, (idx - k) % N] = 1.0
    return R


def is_permutation_matrix(R: torch.Tensor) -> bool:
    """
    Quick check for permutation matrix (dense).
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    N = R.shape[0]
    if not torch.all((R == 0) | (R == 1)):
        return False
    if not torch.all(R.sum(dim=0) == 1):
        return False
    if not torch.all(R.sum(dim=1) == 1):
        return False
    # Orthogonality
    I = torch.eye(N, device=R.device, dtype=R.dtype)
    return torch.allclose(R @ R.T, I)


# ---------------------------------------------------------------------
# DFT / IDFT matrices (unitary)
# ---------------------------------------------------------------------
def dft_matrix(N: int, device: Optional[torch.device] = None,
               dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Unitary DFT matrix F with entries:
        F[m,n] = exp(-2π i m n / N) / sqrt(N)

    Implemented via torch.polar to guarantee correct precision:
    - float32 -> complex64
    - float64 -> complex128
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    float_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    n = torch.arange(N, device=device, dtype=float_dtype)
    m = n[:, None]
    angle = (-2.0 * torch.pi * (m * n) / float(N)).to(float_dtype)

    twiddle = torch.polar(torch.ones_like(angle), angle)  # complex64/128 depending on float_dtype
    F = twiddle / torch.sqrt(torch.tensor(float(N), device=device, dtype=float_dtype))
    return F.to(dtype)


def idft_matrix(N: int, device: Optional[torch.device] = None,
                dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Inverse DFT matrix (Hermitian transpose of F).
    """
    F = dft_matrix(N, device=device, dtype=dtype)
    return F.conj().T

def fourier_mode(N: int, m: int, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Normalized Fourier mode v_m[n] = exp(-2π i m n / N) / sqrt(N).
    """
    float_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    n = torch.arange(N, device=device, dtype=float_dtype)
    angle = (-2.0 * torch.pi * (float(m) * n) / float(N)).to(float_dtype)

    vm = torch.polar(torch.ones_like(angle), angle)
    vm = vm / torch.sqrt(torch.tensor(float(N), device=device, dtype=float_dtype))
    return vm.to(dtype)


# ---------------------------------------------------------------------
# Spectral diagonalization check (torch)
# ---------------------------------------------------------------------

def diagonalize_shift(N: int, k: int, device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.complex64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute D = F R_k F* (should be diagonal).
    Returns (D, F).
    """
    Rk = shift_matrix(N, k, device=device, dtype=torch.float32).to(dtype)
    F = dft_matrix(N, device=device, dtype=dtype)
    D = F @ Rk @ F.conj().T
    return D, F

def shift_eigenvalues(N: int, k: int, device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Eigenvalues λ_m = exp(-2π i k m / N), m=0,...,N-1
    """
    float_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    m = torch.arange(N, device=device, dtype=float_dtype)
    angle = (-2.0 * torch.pi * (float(k) * m) / float(N)).to(float_dtype)

    lam = torch.polar(torch.ones_like(angle), angle)
    return lam.to(dtype)

def check_diagonalization(
    N: int,
    k: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
    atol: float = 1e-10,
) -> bool:
    """
    Check that F R_k F* is diagonal and matches the expected eigenvalues.

    Notes
    -----
    Use complex128 for near machine-precision checks (~1e-15).
    For complex64 you typically need looser tolerances (e.g. 1e-5).
    """
    D, _ = diagonalize_shift(N, k, device=device, dtype=dtype)
    diag = torch.diag(D)
    target = shift_eigenvalues(N, k, device=device, dtype=dtype)

    off = D - torch.diag(diag)
    off_max = torch.max(torch.abs(off)).item()

    diag_ok = torch.allclose(diag, target, atol=atol, rtol=0.0)
    off_ok = off_max < atol
    return bool(diag_ok and off_ok)
