"""
BIT 2118: Scientific Computing
Task 19 – Numerical Optimization
Real-World Case: Minimizing Production Cost for a Manufacturing Facility

A factory's daily production cost (in $) depends on the number of units x
produced per hour:

  C(x) = 0.5*x^2 - 20*x + 300

This is a quadratic with a clear minimum. The analytical minimum is at
x = 20 units/hour (vertex of parabola). We use:
  1. Brute force grid search
  2. scipy.optimize.minimize_scalar  (single variable)
  3. scipy.optimize.minimize  (gradient-based, multivariate demo)
  4. scipy.optimize.minimize with constraints (constrained optimization)
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt

# ── 1. Define cost function ───────────────────────────────────────────────────
def cost(x):
    return 0.5 * x**2 - 20 * x + 300

# Exact minimum: dC/dx = x - 20 = 0  → x = 20, C = 100
print("=== Production Cost Optimization ===")
print(f"  C(x) = 0.5x² - 20x + 300")
print(f"  Analytic minimum: x = 20, C(20) = {cost(20)}")

# ── 2. Brute force grid search ────────────────────────────────────────────────
x_grid = np.arange(0, 51, 1.0)
c_grid = cost(x_grid)
x_min_grid = x_grid[np.argmin(c_grid)]
print(f"\n  Grid search minimum : x = {x_min_grid},  C = {cost(x_min_grid):.2f}")

# ── 3. scipy.optimize.minimize_scalar ────────────────────────────────────────
result_sc = minimize_scalar(cost, bounds=(0, 50), method="bounded")
print(f"  minimize_scalar     : x = {result_sc.x:.6f},  C = {result_sc.fun:.6f}")
print(f"  Converged: {result_sc.success}")

# ── 4. Multivariate: 2D cost surface ──────────────────────────────────────────
# C(x, y) = (x - 3)^2 + (y - 7)^2 + 2  → minimum at (3, 7)
def cost2d(xy):
    x, y = xy
    return (x - 3)**2 + (y - 7)**2 + 2

result_2d = minimize(cost2d, x0=[0, 0], method="Nelder-Mead")
print(f"\n  2D minimization (x-3)²+(y-7)²+2:")
print(f"    Minimum at x={result_2d.x[0]:.4f}, y={result_2d.x[1]:.4f}")
print(f"    Minimum value = {result_2d.fun:.6f}  (expected 2.0)")

# ── 5. Constrained optimization ──────────────────────────────────────────────
# Maximize profit: P(x) = 30x - C(x) = 30x - 0.5x² + 20x - 300
# Constraints: x ≥ 5 (minimum run) and x ≤ 40 (machine capacity)
def neg_profit(x):
    # minimize negative profit = maximize profit
    return -(30 * x - cost(x))

from scipy.optimize import minimize as sp_minimize
result_con = sp_minimize(neg_profit, x0=[20], method="SLSQP",
                         bounds=[(5, 40)])
x_opt = result_con.x[0]
print(f"\n  Constrained profit maximization (5 ≤ x ≤ 40):")
print(f"    Optimal production rate: x = {x_opt:.4f} units/hr")
print(f"    Maximum profit: ${-result_con.fun:.2f}")

# ── 6. Experiment – change cost function (steeper quadratic) ─────────────────
def cost_exp(x):
    return 2 * x**2 - 20 * x + 300    # steeper parabola

result_exp = minimize_scalar(cost_exp, bounds=(0, 50), method="bounded")
print(f"\n=== Experiment: Steeper cost (2x² instead of 0.5x²) ===")
print(f"  Optimal x = {result_exp.x:.4f},  C = {result_exp.fun:.4f}")
print(f"  Analytic: x = 5.0,  C = {cost_exp(5)}")
print("  Steeper parabola → optimal quantity shifts toward lower production.")

# ── 7. Visualisation ─────────────────────────────────────────────────────────
x_plot = np.linspace(0, 50, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1D cost curve
axes[0].plot(x_plot, cost(x_plot), color="steelblue", label="C(x)")
axes[0].axvline(x=result_sc.x, color="red", linestyle="--",
                label=f"Min at x={result_sc.x:.1f}")
axes[0].scatter([result_sc.x], [result_sc.fun], color="red", zorder=5, s=60)
axes[0].set_title("Production Cost  C(x) = 0.5x² − 20x + 300")
axes[0].set_xlabel("Production rate x (units/hr)")
axes[0].set_ylabel("Daily Cost ($)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# 2D cost surface contour
x2 = np.linspace(-2, 8, 100)
y2 = np.linspace(2, 12, 100)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = (X2 - 3)**2 + (Y2 - 7)**2 + 2
cs = axes[1].contourf(X2, Y2, Z2, levels=20, cmap="YlOrRd")
axes[1].plot(result_2d.x[0], result_2d.x[1], "b*", markersize=12, label="Minimum")
axes[1].set_title("2D Cost Surface  (x−3)²+(y−7)²+2")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
plt.colorbar(cs, ax=axes[1], label="Cost")

plt.tight_layout()
plt.savefig("results/Task_19_optimization.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_19_optimization.png")
