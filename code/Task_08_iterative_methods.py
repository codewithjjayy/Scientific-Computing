"""
BIT 2118: Scientific Computing
Task 08 – Iterative Methods for Linear Systems
Real-World Case: Temperature Distribution in a Metal Rod (Jacobi & Gauss-Seidel)

A metal rod is divided into 4 interior points. The temperatures at the two
ends are fixed (boundary conditions). We want to find the steady-state
temperature at the interior points.

This gives a tridiagonal linear system that we solve iteratively using:
  1. Jacobi Method
  2. Gauss-Seidel Method

Both methods start from an initial guess and improve it repeatedly until
the solution is close enough (convergence tolerance = 1e-6).

Boundary conditions:  T0 = 100°C (left end),  T5 = 50°C (right end)
Interior: T1, T2, T3, T4 (unknowns)

Equations (heat balance, uniform spacing):
  T0 - 2*T1 + T2 = 0
  T1 - 2*T2 + T3 = 0
  T2 - 2*T3 + T4 = 0
  T3 - 2*T4 + T5 = 0
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the system 
A = np.array([
    [-2,  1,  0,  0],
    [ 1, -2,  1,  0],
    [ 0,  1, -2,  1],
    [ 0,  0,  1, -2]
], dtype=float)

T0, T5 = 100.0, 50.0
b = np.array([-T0, 0, 0, -T5], dtype=float)

# Exact solution for reference
x_exact = np.linalg.solve(A, b)
print("=== Temperature Distribution System ===")
print(f"  Exact (np.linalg.solve): {np.round(x_exact, 4)}")

#  2. Jacobi Method 
def jacobi(A, b, tol=1e-6, max_iter=200):
    n = len(b)
    x = np.zeros(n)          # initial guess = all zeros
    history = [x.copy()]

    for iteration in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        history.append(x_new.copy())
        if np.max(np.abs(x_new - x)) < tol:
            print(f"  Jacobi converged in {iteration + 1} iterations.")
            return x_new, history
        x = x_new

    print("  Jacobi did not converge.")
    return x, history

#  3. Gauss-Seidel Method 
def gauss_seidel(A, b, tol=1e-6, max_iter=200):
    n = len(b)
    x = np.zeros(n)
    history = [x.copy()]

    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        history.append(x.copy())
        if np.max(np.abs(x - x_old)) < tol:
            print(f"  Gauss-Seidel converged in {iteration + 1} iterations.")
            return x, history

    print("  Gauss-Seidel did not converge.")
    return x, history

print("\n=== Iterative Solutions ===")
x_jac, hist_jac = jacobi(A, b)
x_gs,  hist_gs  = gauss_seidel(A, b)

print(f"\n  Jacobi result      : {np.round(x_jac, 4)}")
print(f"  Gauss-Seidel result: {np.round(x_gs, 4)}")
print(f"  Exact solution     : {np.round(x_exact, 4)}")

#  4. Experiment – different boundary conditions 
T0_new, T5_new = 200.0, 20.0
b_exp = np.array([-T0_new, 0, 0, -T5_new], dtype=float)
x_exp, _ = gauss_seidel(A, b_exp)
print(f"\n=== Experiment: T0={T0_new}°C, T5={T5_new}°C ===")
print(f"  T1={x_exp[0]:.2f}, T2={x_exp[1]:.2f}, T3={x_exp[2]:.2f}, T4={x_exp[3]:.2f}")
print("  Steeper gradient → temperatures drop more sharply across the rod.")

# 5. Visualisation 
# Temperature profile along the rod
positions_orig = [T0] + list(x_jac) + [T5]
positions_exp  = [T0_new] + list(x_exp) + [T5_new]
x_pos = [0, 1, 2, 3, 4, 5]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x_pos, positions_orig, "o-", color="crimson", label="Original (100→50°C)")
axes[0].plot(x_pos, positions_exp,  "s--", color="steelblue", label="Experiment (200→20°C)")
axes[0].set_title("Steady-State Temperature Profile")
axes[0].set_xlabel("Rod Position")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Convergence: max change per iteration
def convergence_history(hist):
    diffs = []
    for i in range(1, len(hist)):
        diffs.append(np.max(np.abs(np.array(hist[i]) - np.array(hist[i-1]))))
    return diffs

jac_conv = convergence_history(hist_jac)
gs_conv  = convergence_history(hist_gs)

axes[1].semilogy(jac_conv, label="Jacobi",        color="orange")
axes[1].semilogy(gs_conv,  label="Gauss-Seidel",  color="green")
axes[1].set_title("Convergence: Max Change per Iteration")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Max |Δx| (log scale)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_08_iterative_methods.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_08_iterative_methods.png")
