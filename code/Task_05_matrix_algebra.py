"""
BIT 2118: Scientific Computing
Task 05 – Matrix Algebra and Matrix Operations
Real-World Case: Network Traffic Flow (Adjacency Matrix)

A small network has 4 nodes (routers). The connection weights between
routers are stored in a matrix. We practice basic matrix operations:
addition, scalar multiplication, transpose, and matrix multiplication.

Assumptions:
  - Matrix values represent link costs (e.g. latency in ms).
  - Two different network snapshots are compared using matrix addition.
"""

import numpy as np
import matplotlib.pyplot as plt

#  1. Define two 4×4 network cost matrices 
# Rows and columns = routers A, B, C, D
# Value = link cost (ms). 0 means no direct link.

A = np.array([
    [0, 2, 5, 0],
    [2, 0, 3, 7],
    [5, 3, 0, 1],
    [0, 7, 1, 0]
], dtype=float)

B = np.array([
    [0, 1, 4, 3],
    [1, 0, 2, 6],
    [4, 2, 0, 2],
    [3, 6, 2, 0]
], dtype=float)

nodes = ["Router-A", "Router-B", "Router-C", "Router-D"]

print("=== Matrix A (Network Snapshot 1) ===")
print(A)

print("\n=== Matrix B (Network Snapshot 2) ===")
print(B)

#  2. Matrix addition – combined cost view 
C = A + B
print("\n=== A + B (Combined costs) ===")
print(C)

#  3. Scalar multiplication – increase all costs by 20%
A_scaled = 1.2 * A
print("\n=== 1.2 × A  (20% congestion increase) ===")
print(np.round(A_scaled, 2))

#  4. Transpose – reverse the direction of all links─
A_T = A.T
print("\n=== Transpose of A ===")
print(A_T)
print(f"  A is symmetric: {np.allclose(A, A_T)}")   # True for undirected networks

#  5. Matrix multiplication – two-hop reachability
# A @ A gives the sum of products of paths of length 2 between nodes.
A2 = A @ A
print("\n=== A × A  (Two-hop path costs) ===")
print(np.round(A2, 2))

#  6. Determinant and inverse 
# Use a non-singular sub-matrix (3×3) for the demo
M = np.array([
    [2, 1, 0],
    [1, 3, 1],
    [0, 1, 4]
], dtype=float)

det_M = np.linalg.det(M)
inv_M = np.linalg.inv(M)
print(f"\n=== 3×3 Sub-matrix M ===")
print(M)
print(f"  Determinant : {det_M:.4f}")
print(f"  Inverse M⁻¹ :")
print(np.round(inv_M, 4))

# Verify: M @ M⁻¹ should equal the identity matrix
identity_check = np.round(M @ inv_M, 10)
print(f"  M × M⁻¹ ≈ I : {np.allclose(identity_check, np.eye(3))}")

# 7. Experiment – change scalar factor to 1.5 
print("\n=== Experiment: 1.5 × A (50% congestion increase) ===")
print(np.round(1.5 * A, 2))
print("  Higher scalar → all link costs increase proportionally.")

#  8. Visualisation – heat-map of matrix A 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, mat, title in zip(axes, [A, C], ["Matrix A – Network Snapshot 1", "A + B – Combined Costs"]):
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(4)); ax.set_xticklabels(nodes, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(nodes)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="Cost (ms)")

plt.tight_layout()
plt.savefig("results/Task_05_matrix_algebra.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_05_matrix_algebra.png")
