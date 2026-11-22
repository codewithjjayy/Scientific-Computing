"""
BIT 2118: Scientific Computing
Task 09 – Eigenvalues and Eigenvectors
Real-World Case: Vibration Modes of a Two-Storey Building

A simplified structural model of a 2-storey building can be represented
by a stiffness matrix. The eigenvalues of this matrix correspond to the
natural frequencies of vibration. The eigenvectors describe the shape
of each vibration mode (how floors move relative to each other).

Engineers use this to make sure a building's natural frequency does not
match common earthquake or wind frequencies (resonance avoidance).

Stiffness matrix (2×2 simplified model):
  K = [[k1 + k2,  -k2],
       [-k2,       k2]]

where k1 = 3000 N/m (ground floor stiffness)
      k2 = 2000 N/m (upper floor stiffness)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Build the stiffness matrix 
k1 = 3000.0   # N/m – ground-floor spring stiffness
k2 = 2000.0   # N/m – upper-floor spring stiffness

K = np.array([
    [k1 + k2, -k2],
    [-k2,      k2]
], dtype=float)

print("=== Stiffness Matrix K ===")
print(K)

#  2. Compute eigenvalues and eigenvectors
# K is symmetric → use eigh (returns real values, faster and more stable)
eigenvalues, eigenvectors = np.linalg.eigh(K)
# eigh already returns sorted eigenvalues (ascending), no need to sort

print(f"\n=== Eigenvalues ===")
for i, lam in enumerate(eigenvalues):
    print(f"  λ{i+1} = {lam:.2f}  →  natural frequency ω{i+1} = √λ{i+1} = {np.sqrt(lam):.4f} rad/s")

print(f"\n=== Eigenvectors (columns = mode shapes) ===")
print(np.round(eigenvectors, 4))
print("  Mode 1 (column 0): both floors move in the same direction (in-phase)")
print("  Mode 2 (column 1): floors move in opposite directions (out-of-phase)")

#  3. Verify: K · v = λ · v 
print("\n=== Verification: K · v = λ · v ===")
for i in range(2):
    lhs = K @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"  Mode {i+1}: max error = {np.max(np.abs(lhs - rhs)):.2e}")

#  4. Larger example – 4×4 tridiagonal (4-storey building)
k = 2500.0
K4 = np.array([
    [2*k,  -k,   0,   0],
    [ -k, 2*k,  -k,   0],
    [  0,  -k, 2*k,  -k],
    [  0,   0,  -k,   k]
], dtype=float)

vals4, vecs4 = np.linalg.eigh(K4)
# eigh returns sorted ascending
print(f"\n=== 4-Storey Building – Natural Frequencies ===")
for i, lam in enumerate(vals4):
    print(f"  Mode {i+1}: ω = {np.sqrt(np.abs(lam)):.4f} rad/s")

# 5. Experiment – increase k1 (stiffer ground floor) 
k1_new = 6000.0
K_exp  = np.array([[k1_new + k2, -k2], [-k2, k2]], dtype=float)
vals_exp, _ = np.linalg.eigh(K_exp)
# eigh returns sorted ascending
print(f"\n=== Experiment: k1 doubled to {k1_new} N/m ===")
for i, lam in enumerate(vals_exp):
    print(f"  λ{i+1} = {lam:.2f}  →  ω{i+1} = {np.sqrt(lam):.4f} rad/s")
print("  Higher stiffness → higher natural frequencies → less risk of resonance.")

# 6. Visualisation – mode shapes 
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
floors = [0, 1, 2]   # ground (0), floor 1, floor 2

for i in range(2):
    mode = [0] + list(eigenvectors[:, i])    # ground node is fixed (0)
    axes[i].plot(mode, floors, "o-", linewidth=2)
    axes[i].set_title(f"Mode {i+1}  (ω = {np.sqrt(eigenvalues[i]):.2f} rad/s)")
    axes[i].set_xlabel("Relative Displacement")
    axes[i].set_ylabel("Floor Level")
    axes[i].axvline(0, color="gray", linestyle="--")
    axes[i].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Building Vibration Mode Shapes")
plt.tight_layout()
plt.savefig("results/Task_09_eigenvalues.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_09_eigenvalues.png")
