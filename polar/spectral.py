"""
polar.spectral
===============

Spectral properties of circular shift operators.

The shift operators {R_k} are diagonalized by the Discrete Fourier Transform (DFT).
Their eigenvectors coincide with the Fourier modes, and the eigenvalues are roots
of unity.
"""

from __future__ import annotations
import numpy as np

from polar.operators import shift_operator


# ---------------------------------------------------------------------
# Discrete Fourier Transform
# ---------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    """
    Unitary Discrete Fourier Transform (DFT) matrix.

    F[m, n] = exp(-2π i m n / N) / sqrt(N)
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    n = np.arange(N)
    m = n[:, None]
    return np.exp(-2j * np.pi * m * n / N) / np.sqrt(N)


def idft_matrix(N: int) -> np.ndarray:
    """
    Inverse DFT matrix (Hermitian transpose of F).
    """
    F = dft_matrix(N)
    return np.conjugate(F).T


# ---------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------

def diagonalize_shift(N: int, k: int):
    """
    Diagonalize the shift operator R_k via the DFT.

    Returns
    -------
    D : ndarray
        Diagonal matrix of eigenvalues.
    F : ndarray
        DFT matrix.
    """
    Rk = shift_operator(N, k)
    F = dft_matrix(N)
    D = F @ Rk @ np.conjugate(F).T
    return D, F


def shift_eigenvalues(N: int, k: int) -> np.ndarray:
    """
    Eigenvalues of the shift operator R_k.

    λ_m = exp(-2π i k m / N),  m = 0,...,N-1
    """
    m = np.arange(N)
    return np.exp(-2j * np.pi * k * m / N)


def shift_eigenvectors(N: int) -> np.ndarray:
    """
    Eigenvectors of all shift operators.

    The columns are the Fourier modes (DFT basis).
    """
    return dft_matrix(N)


# ---------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------

def check_diagonalization(N: int, k: int, atol: float = 1e-10) -> bool:
    """
    Check that F R_k F* is diagonal and matches the expected eigenvalues.
    """
    D, _ = diagonalize_shift(N, k)
    diag = np.diag(D)
    target = shift_eigenvalues(N, k)

    off_diag = D - np.diag(diag)

    return (
        np.allclose(off_diag, 0.0, atol=atol) and
        np.allclose(diag, target, atol=atol)
    )


def fourier_mode(N: int, m: int) -> np.ndarray:
    """
    Normalized Fourier mode v_m[n] = exp(-2π i m n / N) / sqrt(N).
    """
    n = np.arange(N)
    return np.exp(-2j * np.pi * m * n / N) / np.sqrt(N)
