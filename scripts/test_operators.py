import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from polar.operators import (
    shift_operator,
    inverse_shift_operator,
    rotation_group,
    compose,
    is_unitary,
    is_permutation,
    apply_shift,
    check_group_axioms,
)


def print_matrix(name: str, M: np.ndarray) -> None:
    print(f"\n{name} (shape={M.shape})")
    print(M)


def main():
    np.set_printoptions(linewidth=140, suppress=True)

    N = 8
    k = 3

    print("=" * 80)
    print("SMOKE TEST: polar.operators")
    print(f"N={N}, k={k}")

    # 1) Build operator
    Rk = shift_operator(N, k)
    print_matrix("R_k", Rk)

    # 2) Inverse operator
    Rinv = inverse_shift_operator(N, k)
    print_matrix("R_-k", Rinv)

    # 3) Composition check: Rk @ R_-k = I
    I = np.eye(N)
    comp = Rk @ Rinv
    print_matrix("R_k @ R_-k", comp)
    print("Allclose to I:", np.allclose(comp, I))

    # 4) Unitary + permutation checks
    print("\nProperties:")
    print("is_unitary(Rk):", is_unitary(Rk))
    print("is_permutation(Rk):", is_permutation(Rk))

    # 5) Group generation
    G = rotation_group(N)
    print("\nGroup size:", len(G))
    print("Check group axioms:", check_group_axioms(N))

    # 6) Compose: R_a R_b = R_{a+b}
    a, b = 2, 5
    Ra = shift_operator(N, a)
    Rb = shift_operator(N, b)
    Rab = compose(Ra, Rb)
    R_aplusb = shift_operator(N, a + b)
    print("\nComposition law:")
    print(f"a={a}, b={b}")
    print("Allclose(Ra@Rb, R_{a+b}):", np.allclose(Rab, R_aplusb))

    # 7) Action on signals: matrix vs direct roll
    x = np.arange(N)
    y_mat = Rk @ x
    y_roll = apply_shift(x, k)

    print("\nSignal action:")
    print("x:", x)
    print("Rk @ x:", y_mat)
    print("apply_shift(x, k):", y_roll)
    print("Matrix action equals apply_shift:", np.allclose(y_mat, y_roll))

    print("\nDONE")


if __name__ == "__main__":
    main()
