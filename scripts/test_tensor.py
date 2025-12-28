import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from polar.tensor import (
    apply_shift_1d,
    shift_matrix,
    dft_matrix,
    fourier_mode,
    diagonalize_shift,
    shift_eigenvalues,
    check_diagonalization,
)


def run_case(N: int, k: int, device: torch.device, cdtype: torch.dtype, atol: float) -> None:
    """
    Run one numerical case with a given complex dtype and tolerance.
    """
    fdtype = torch.float64 if cdtype == torch.complex128 else torch.float32

    print("\n" + "-" * 80)
    print(f"CASE: dtype={cdtype} (float part {fdtype}), atol={atol:g}")

    # 1) Shift action vs matrix action (real)
    x = torch.arange(N, device=device, dtype=fdtype)
    Rk = shift_matrix(N, k, device=device, dtype=fdtype)

    y_roll = apply_shift_1d(x, k)
    y_mat = Rk @ x

    print("\nSignal action:")
    print("x:", x)
    print("apply_shift_1d(x,k):", y_roll)
    print("Rk @ x:", y_mat)
    print("match:", torch.allclose(y_roll, y_mat))

    # 2) Fourier mode vs DFT row (complex)
    F = dft_matrix(N, device=device, dtype=cdtype)
    print("\nFourier mode checks: max ||row(F)-v_m||")
    max_row_err = 0.0
    for m in range(N):
        vm = fourier_mode(N, m, device=device, dtype=cdtype)
        row = F[m, :]
        err = torch.linalg.norm(row - vm).item()
        max_row_err = max(max_row_err, err)
    print(f"max row error: {max_row_err:.3e}")

    # 3) Diagonalization
    D, _ = diagonalize_shift(N, k, device=device, dtype=cdtype)
    diag = torch.diag(D)
    target = shift_eigenvalues(N, k, device=device, dtype=cdtype)

    # eigenvalue error metrics
    ev_abs_err = torch.max(torch.abs(diag - target)).item()

    off = D - torch.diag(diag)
    off_max = torch.max(torch.abs(off)).item()
    off_fro = torch.linalg.norm(off).item()
    D_fro = torch.linalg.norm(D).item()
    off_rel = off_fro / (D_fro + 1e-30)

    print("\nSpectral diagnostics:")
    print(f"max |diag-target|:      {ev_abs_err:.3e}")
    print(f"max |off-diagonal|:     {off_max:.3e}")
    print(f"||off||_F:              {off_fro:.3e}")
    print(f"||off||_F / ||D||_F:    {off_rel:.3e}")

    # pass/fail checks (tolerance-aware)
    ev_ok = torch.allclose(diag, target, atol=atol, rtol=0.0)
    diag_ok = check_diagonalization(N, k, device=device, dtype=cdtype, atol=atol)

    print("\nChecks:")
    print("Eigenvalues match (allclose):", bool(ev_ok))
    print("Check diagonalization:", bool(diag_ok))


def main():
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)

    device = torch.device("cpu")
    N = 8
    k = 3

    print("=" * 80)
    print("SMOKE TEST: polar.tensor (numerically stable)")
    print(f"device={device}, N={N}, k={k}")

    # complex64: looser tol (single-precision complex matmuls accumulate error)
    run_case(N, k, device, torch.complex64, atol=1e-5)

    # complex128: tight tol (near machine precision)
    run_case(N, k, device, torch.complex128, atol=1e-10)

    print("\nDONE")


if __name__ == "__main__":
    main()
