"""
BIT 2118: Scientific Computing
Task 10 – Root Finding: Bisection and Fixed-Point Methods
Real-World Case: Finding the Break-Even Point of a Business

A small business has monthly costs and revenue described by:
  Revenue:  R(x) = 50*x          ($50 per unit sold)
  Cost:     C(x) = 0.5*x^2 + 10  ($0.5 per unit^2 + $10 fixed)

Profit = Revenue - Cost = 0 at the break-even point.

We want to find x (units) where:
  f(x) = 50*x - 0.5*x^2 - 10 = 0

We solve this using:
  1. Bisection Method
  2. Fixed-Point Iteration
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Define the function ────────────────────────────────────────────────────
def f(x):
    return 50 * x - 0.5 * x**2 - 10

# ── 2. Bisection Method ───────────────────────────────────────────────────────
def bisection(func, a, b, tol=1e-6, max_iter=100):
    """
    Find root of func in [a, b].
    Requires f(a) and f(b) to have opposite signs.
    """
    if func(a) * func(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    history = []
    for i in range(max_iter):
        mid = (a + b) / 2
        history.append(mid)

        if abs(func(mid)) < tol or (b - a) / 2 < tol:
            print(f"  Bisection converged in {i + 1} iterations.")
            return mid, history

        if func(a) * func(mid) < 0:
            b = mid
        else:
            a = mid

    return (a + b) / 2, history

# Two roots exist:  small root ≈ 0.2  and  large (meaningful) root ≈ 99.6
# We find the larger root – this is where the business moves from loss to profit
print("=== Break-Even Root Finding ===")
root_b, hist_b = bisection(f, a=50, b=100)
print(f"  Bisection root   : x = {root_b:.6f} units")
print(f"  f(root) = {f(root_b):.2e}")

# ── 3. Fixed-Point Iteration ──────────────────────────────────────────────────
# Rearrange f(x) = 0 as x = g(x):
#   50*x - 0.5*x^2 - 10 = 0
#   50*x = 0.5*x^2 + 10
#   x = (0.5*x^2 + 10) / 50   → g(x)
def g(x):
    return (0.5 * x**2 + 10) / 50

def fixed_point(gfunc, x0, tol=1e-6, max_iter=200):
    x = x0
    history = [x]
    for i in range(max_iter):
        x_new = gfunc(x)
        history.append(x_new)
        if abs(x_new - x) < tol:
            print(f"  Fixed-Point converged in {i + 1} iterations.")
            return x_new, history
        x = x_new
    print("  Fixed-Point did not converge (try a different g(x)).")
    return x, history

# Fixed-point converges to the small root (~0.2); bisection found the larger one
root_fp, hist_fp = fixed_point(g, x0=0.2)
print(f"  Fixed-Point root : x = {root_fp:.6f} units  (small root; g(x) converges here)")
print(f"  f(root) = {f(root_fp):.2e}")
print(f"  Note: bisection found the economically meaningful root x ≈ {root_b:.2f}")

# ── 4. Experiment – change function (higher fixed cost) ──────────────────────
def f2(x):
    return 50 * x - 0.5 * x**2 - 50   # fixed cost raised to $50

root_b2, _ = bisection(f2, a=50, b=100)
print(f"\n=== Experiment: Fixed cost raised to $50 ===")
print(f"  Break-even shifts to x = {root_b2:.4f} units")
print("  Higher fixed cost → more units needed to break even.")

# ── 5. Visualisation ─────────────────────────────────────────────────────────
x_vals = np.linspace(0, 100, 400)
y_vals = f(x_vals)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: function plot with roots
axes[0].plot(x_vals, y_vals, color="steelblue", label="f(x) = Profit")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axvline(root_b,  color="red",   linestyle="--", label=f"Root ≈ {root_b:.3f}")
axes[0].set_title("Break-Even Analysis – Profit Function")
axes[0].set_xlabel("Units sold (x)")
axes[0].set_ylabel("Profit ($)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].set_ylim(-50, 150)

# Right: bisection convergence
axes[1].plot(range(len(hist_b)), hist_b, "o-", color="crimson", markersize=4)
axes[1].axhline(root_b, color="green", linestyle="--", label="True root")
axes[1].set_title("Bisection Method Convergence")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Midpoint estimate")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_10_bisection_fixedpoint.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_10_bisection_fixedpoint.png")
