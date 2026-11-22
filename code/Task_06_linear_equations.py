"""
BIT 2118: Scientific Computing
Task 06 – Solution of Systems of Linear Equations
Real-World Case: Electrical Circuit – Finding Branch Currents (Kirchhoff's Laws)

A simple circuit has 3 unknown branch currents: I1, I2, I3 (in Amperes).
Applying Kirchhoff's Voltage Law (KVL) at three loops gives 3 equations.

System:
  2*I1 +  I2        =  5     (loop 1)
   I1  + 3*I2 +  I3 = 10     (loop 2)
         2*I2 + 4*I3 =  8    (loop 3)

We solve this system three ways:
  1. np.linalg.solve  (direct NumPy solver)
  2. Cramer's Rule    (determinant-based)
  3. Matrix inverse   (I = A⁻¹ b)
"""

import numpy as np
import matplotlib.pyplot as plt

#  1. Define the coefficient matrix A and RHS vector b
A = np.array([
    [2, 1, 0],
    [1, 3, 1],
    [0, 2, 4]
], dtype=float)

b = np.array([5, 10, 8], dtype=float)

print("=== Circuit Equations  (A · x = b) ===")
print(f"  A =\n{A}")
print(f"  b = {b}")

#  2. NumPy direct solver 
x_np = np.linalg.solve(A, b)
print(f"\n=== Solution via np.linalg.solve ===")
print(f"  I1 = {x_np[0]:.4f} A")
print(f"  I2 = {x_np[1]:.4f} A")
print(f"  I3 = {x_np[2]:.4f} A")

# Verify: A @ x should equal b
residual = np.max(np.abs(A @ x_np - b))
print(f"  Residual (max |A·x - b|): {residual:.2e}")

#  3. Cramer's Rule 
det_A = np.linalg.det(A)

def cramers(A, b, j):
    """Replace column j of A with b, return det ratio."""
    A_j = A.copy()
    A_j[:, j] = b
    return np.linalg.det(A_j) / det_A

I1_c = cramers(A, b, 0)
I2_c = cramers(A, b, 1)
I3_c = cramers(A, b, 2)

print(f"\n=== Cramer's Rule ===")
print(f"  det(A) = {det_A:.4f}")
print(f"  I1 = {I1_c:.4f} A")
print(f"  I2 = {I2_c:.4f} A")
print(f"  I3 = {I3_c:.4f} A")

#  4. Matrix inverse method: x = A⁻¹ · b 
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b
print(f"\n=== Inverse Method  (x = A⁻¹ · b) ===")
print(f"  I1 = {x_inv[0]:.4f} A")
print(f"  I2 = {x_inv[1]:.4f} A")
print(f"  I3 = {x_inv[2]:.4f} A")

#  5. Experiment – change b (different voltage source) 
b2 = np.array([8, 12, 6], dtype=float)
x2 = np.linalg.solve(A, b2)
print(f"\n=== Experiment: b changed to {b2} ===")
print(f"  I1 = {x2[0]:.4f} A,  I2 = {x2[1]:.4f} A,  I3 = {x2[2]:.4f} A")
print("  Higher voltage source → currents redistribute across all branches.")

#  6. Visualisation – bar chart comparing currents 
labels = ["I1", "I2", "I3"]
x_vals_orig = x_np
x_vals_exp  = x2

x_pos = np.arange(3)
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x_pos - width/2, x_vals_orig, width, label="Original b", color="steelblue")
ax.bar(x_pos + width/2, x_vals_exp,  width, label="Experiment b", color="salmon")
ax.set_title("Branch Currents – Original vs Experiment")
ax.set_xlabel("Current")
ax.set_ylabel("Amperes (A)")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.axhline(0, color="black", linewidth=0.8)
for i, (v1, v2) in enumerate(zip(x_vals_orig, x_vals_exp)):
    ax.text(i - width/2, v1 + 0.05, f"{v1:.2f}", ha="center", fontsize=8)
    ax.text(i + width/2, v2 + 0.05, f"{v2:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("results/Task_06_linear_equations.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_06_linear_equations.png")
