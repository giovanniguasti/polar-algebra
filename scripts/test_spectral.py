import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



import numpy as np

from polar.spectral import (
    dft_matrix,
    diagonalize_shift,
    shift_eigenvalues,
    check_diagonalization,
    fourier_mode,
)


def print_matrix(name, M):
    print(f"\n{name} (shape={M.shape})")
    print(M)


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    N = 8
    k = 3

    print("=" * 80)
    print("SMOKE TEST: polar.spectral")
    print(f"N={N}, k={k}")

    # 1) DFT matrix
    F = dft_matrix(N)
    print_matrix("DFT matrix F", F)

    # 1b) Compare Fourier modes with rows of F and columns of F*
    print("\nFourier mode checks (normalized):")
    for m in range(N):
        vm = fourier_mode(N, m)

        row_m = F[m, :]                  # v_m
        col_m = np.conjugate(F[m, :])    # (F*)[:, m] = conj(v_m)

        ok_row = np.allclose(row_m, vm)
        ok_col = np.allclose(col_m, np.conjugate(vm))

        err_row = np.linalg.norm(row_m - vm)
        err_col = np.linalg.norm(col_m - np.conjugate(vm))

        print(f"m={m:2d} | row(F)[m]==v_m: {ok_row} (||·||={err_row:.2e}) "
              f"| col(F*)[m]==conj(v_m): {ok_col} (||·||={err_col:.2e})")



    # 2) Diagonalization
    D, _ = diagonalize_shift(N, k)
    print_matrix("F R_k F*", D)

    off = D - np.diag(np.diag(D))
    off_max = np.max(np.abs(off))
    off_fro = np.linalg.norm(off, ord="fro")
    D_fro = np.linalg.norm(D, ord="fro")

    print("\nOff-diagonal diagnostics:")
    print("Max |off-diagonal|:", off_max)
    print("Frobenius ||off||_F:", off_fro)
    print("Relative ||off||_F / ||D||_F:", off_fro / (D_fro + 1e-30))

    # 3) Extract diagonal (eigenvalues)
    diag = np.diag(D)
    print("\nExtracted eigenvalues:")
    print(diag)

    # 4) Theoretical eigenvalues
    target = shift_eigenvalues(N, k)
    print("\nTheoretical eigenvalues:")
    print(target)

    print("\nEigenvalues match:",
          np.allclose(diag, target))

    # 5) Global consistency check
    print("\nCheck diagonalization:", check_diagonalization(N, k))

    print("\nDONE")


if __name__ == "__main__":
    main()
