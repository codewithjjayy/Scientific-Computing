"""
BIT 2118: Scientific Computing
Task 07 – Gaussian Elimination and LU Decomposition
Real-World Case: Structural Load Analysis – Beam Support Forces

A simply-supported beam has 3 unknown reaction forces: R1, R2, R3 (kN).
Applying equilibrium equations gives:

   R1 +   R2         =  20   (sum of vertical forces)
  2*R1 + 5*R2 + 3*R3 =  70   (sum of moments about left support)
        3*R2  + 4*R3 =  36   (second moment equation)

We solve this using:
  1. Manual Gaussian Elimination (coded step by step).
  2. LU Decomposition via scipy.linalg.lu.
"""

import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
import matplotlib.pyplot as plt

# ── 1. Define the system ──────────────────────────────────────────────────────
A = np.array([
    [1, 1, 0],
    [2, 5, 3],
    [0, 3, 4]
], dtype=float)

b = np.array([20, 70, 36], dtype=float)

print("=== Beam Support Force System  A · x = b ===")
print(f"A =\n{A}\nb = {b}")

# ── 2. Manual Gaussian Elimination ───────────────────────────────────────────
def gaussian_elimination(A, b):
    n = len(b)
    # Build augmented matrix
    M = np.hstack([A.copy(), b.copy().reshape(-1, 1)])

    # Forward elimination
    for col in range(n):
        # Partial pivoting – swap rows so the largest value is on the diagonal
        max_row = col + np.argmax(np.abs(M[col:, col]))
        M[[col, max_row]] = M[[max_row, col]]

        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row] = M[row] - factor * M[col]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]

    return x

x_gauss = gaussian_elimination(A, b)
print("\n=== Gaussian Elimination Result ===")
print(f"  R1 = {x_gauss[0]:.4f} kN")
print(f"  R2 = {x_gauss[1]:.4f} kN")
print(f"  R3 = {x_gauss[2]:.4f} kN")

# Verify
print(f"  Residual: {np.max(np.abs(A @ x_gauss - b)):.2e}")

# ── 3. LU Decomposition via SciPy ─────────────────────────────────────────────
# LU decomposes A = P · L · U
# P = permutation, L = lower triangular, U = upper triangular
P, L, U = lu(A)
print("\n=== LU Decomposition (SciPy) ===")
print(f"  L (lower) =\n{np.round(L, 4)}")
print(f"  U (upper) =\n{np.round(U, 4)}")

lu_fac = lu_factor(A)
x_lu   = lu_solve(lu_fac, b)
print(f"\n  LU solve result:")
print(f"  R1 = {x_lu[0]:.4f} kN,  R2 = {x_lu[1]:.4f} kN,  R3 = {x_lu[2]:.4f} kN")
print(f"  Residual: {np.max(np.abs(A @ x_lu - b)):.2e}")

# ── 4. Experiment – change load vector ───────────────────────────────────────
b2 = np.array([30, 90, 48], dtype=float)
x2 = lu_solve(lu_factor(A), b2)
print(f"\n=== Experiment: b changed to {b2} (heavier load) ===")
print(f"  R1 = {x2[0]:.4f} kN,  R2 = {x2[1]:.4f} kN,  R3 = {x2[2]:.4f} kN")
print("  Larger loads → larger reaction forces, as expected.")

# ── 5. Visualisation – triangular structure of L and U ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, title in zip(axes, [L, U], ["L (Lower Triangular)", "U (Upper Triangular)"]):
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-5, vmax=5)
    ax.set_title(title)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)
plt.suptitle("LU Decomposition of Beam Stiffness Matrix")
plt.tight_layout()
plt.savefig("results/Task_07_lu_decomposition.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_07_lu_decomposition.png")
