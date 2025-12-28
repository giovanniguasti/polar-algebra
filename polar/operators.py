"""
polar.operators
================

Discrete circular shift (rotation) operators acting on angular signals.

The operators {R_k} form a unitary representation of the cyclic group Z_N.
They act on vectors x ∈ C^N as:

    (R_k x)[n] = x[(n - k) mod N]

and are represented by permutation matrices.
"""

from __future__ import annotations
import numpy as np
from typing import List


# ---------------------------------------------------------------------
# Core operators
# ---------------------------------------------------------------------

def shift_operator(N: int, k: int) -> np.ndarray:
    """
    Circular shift operator R_k such that (R_k x)[n] = x[(n - k) mod N].
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    k = k % N

    R = np.zeros((N, N), dtype=float)
    for n in range(N):
        R[n, (n - k) % N] = 1.0
    return R


def inverse_shift_operator(N: int, k: int) -> np.ndarray:
    """
    Inverse circular shift operator R_{-k}.
    """
    return shift_operator(N, -k)


# ---------------------------------------------------------------------
# Group structure
# ---------------------------------------------------------------------

def rotation_group(N: int) -> List[np.ndarray]:
    """
    Generate the cyclic group {R_k}_{k=0,...,N-1}.

    Parameters
    ----------
    N : int
        Signal length.

    Returns
    -------
    group : list of ndarray
        List of N shift operators.
    """
    return [shift_operator(N, k) for k in range(N)]


def compose(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """
    Composition of two shift operators.

    For shifts R_a and R_b, this corresponds to R_{a+b}.
    """
    return Ra @ Rb


# ---------------------------------------------------------------------
# Algebraic properties (sanity checks)
# ---------------------------------------------------------------------

def is_unitary(R: np.ndarray, atol: float = 1e-10) -> bool:
    """
    Check whether an operator is unitary (R* R = I).
    """
    I = np.eye(R.shape[0])
    return np.allclose(R.conj().T @ R, I, atol=atol)


def is_permutation(R: np.ndarray) -> bool:
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False

    if not np.all((R == 0.0) | (R == 1.0)):
        return False

    if not np.all(R.sum(axis=0) == 1):
        return False

    if not np.all(R.sum(axis=1) == 1):
        return False

    return True



# ---------------------------------------------------------------------
# Action on signals
# ---------------------------------------------------------------------

def apply_shift(x: np.ndarray, k: int) -> np.ndarray:
    """
    Apply circular shift directly to a signal (without building R_k).
    (R_k x)[n] = x[(n - k) mod N]

    Parameters
    ----------
    x : ndarray of shape (N,)
        Input signal.
    k : int
        Shift index.

    Returns
    -------
    y : ndarray
        Shifted signal.
    """
    N = x.shape[0]
    return np.roll(x, shift=k)


# ---------------------------------------------------------------------
# Convenience / debugging
# ---------------------------------------------------------------------

def check_group_axioms(N: int, atol: float = 1e-10) -> bool:
    """
    Verify group axioms for the rotation group Z_N.

    Checks:
    - closure
    - identity
    - inverse
    - associativity (implicit via matrix product)
    """
    group = rotation_group(N)

    # identity
    I = shift_operator(N, 0)
    if not np.allclose(group[0], I, atol=atol):
        return False

    # inverses
    for k in range(N):
        Rk = shift_operator(N, k)
        Rm = shift_operator(N, -k)
        if not np.allclose(Rk @ Rm, I, atol=atol):
            return False

    return True
