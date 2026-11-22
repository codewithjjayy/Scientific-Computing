"""

Task 11 – Root Finding: Newton-Raphson and Secant Methods
Real-World Case: Optimal Pipe Radius for a Water Supply Network

A water pipe's flow rate is modelled by a nonlinear equation.
The engineer needs to find the pipe radius r (metres) that delivers
exactly Q = 0.05 m³/s (5 litres/second) under a fixed pressure.

Hagen-Poiseuille approximation (simplified):
  Q(r) = C * r^4   where C = π * ΔP / (8 * μ * L)

Using:  ΔP = 10 000 Pa, μ = 0.001 Pa·s, L = 50 m, π ≈ 3.14159
  C = π * 10000 / (8 * 0.001 * 50) = ~7853.98

So:  f(r) = 7853.98 * r^4 - 0.05 = 0
     f'(r) = 4 * 7853.98 * r^3
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Constants and function definitions
C = np.pi * 10000 / (8 * 0.001 * 50)   # ≈ 7853.98
Q_target = 0.05                          # m³/s

def f(r):
    return C * r**4 - Q_target

def df(r):
    return 4 * C * r**3

print(f"=== Pipe Radius Root Finding ===")
print(f"  C = {C:.4f},  Q_target = {Q_target}")

# 2. Newton-Raphson Method 
# x_{n+1} = x_n - f(x_n) / f'(x_n)
def newton_raphson(func, dfunc, x0, tol=1e-8, max_iter=50):
    x = x0
    history = [x]
    for i in range(max_iter):
        fx  = func(x)
        dfx = dfunc(x)
        if abs(dfx) < 1e-14:
            print("  Derivative too small – Newton-Raphson stopping.")
            break
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < tol:
            print(f"  Newton-Raphson converged in {i + 1} iterations.")
            return x_new, history
        x = x_new
    return x, history

root_nr, hist_nr = newton_raphson(f, df, x0=0.1)
print(f"  Newton-Raphson root : r = {root_nr:.8f} m")
print(f"  Verification: Q = C * r^4 = {C * root_nr**4:.6f} m³/s  (target {Q_target})")

# 3. Secant Method
# Does NOT require the derivative – uses two previous points instead.
# x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
def secant(func, x0, x1, tol=1e-8, max_iter=50):
    history = [x0, x1]
    for i in range(max_iter):
        f0, f1 = func(x0), func(x1)
        if abs(f1 - f0) < 1e-14:
            print("  Denominator too small – Secant stopping.")
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            print(f"  Secant converged in {i + 1} iterations.")
            return x2, history
        x0, x1 = x1, x2
    return x1, history

root_sec, hist_sec = secant(f, x0=0.05, x1=0.15)
print(f"  Secant root         : r = {root_sec:.8f} m")

# 4. Experiment 

def f2(r):
    return C * r**4 - Q_new
root_nr2, _ = newton_raphson(f2, df, x0=0.1)
print(f"\n=== Experiment: Q_target doubled to {Q_new} m³/s ===")
print(f"  New radius : r = {root_nr2:.6f} m  (was {root_nr:.6f} m)")
print("  Larger flow → larger pipe radius required, as expected.")

# 5. Visualisation 
r_vals = np.linspace(0.01, 0.15, 300)
f_vals = f(r_vals)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(r_vals, f_vals, color="steelblue", label="f(r)")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axvline(root_nr, color="red", linestyle="--", label=f"Root ≈ {root_nr:.5f} m")
axes[0].set_title("Flow Equation  f(r) = C·r⁴ − Q")
axes[0].set_xlabel("Pipe radius r (m)")
axes[0].set_ylabel("f(r)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Convergence comparison
iters_nr  = list(range(len(hist_nr)))
iters_sec = list(range(len(hist_sec)))
axes[1].semilogy([abs(h - root_nr) + 1e-15 for h in hist_nr],  "o-", label="Newton-Raphson", color="crimson")
axes[1].semilogy([abs(h - root_nr) + 1e-15 for h in hist_sec], "s--", label="Secant",         color="orange")
axes[1].set_title("Convergence Comparison")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("|error| (log scale)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_11_newton_secant.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_11_newton_secant.png")
