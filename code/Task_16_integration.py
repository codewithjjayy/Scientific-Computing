"""
BIT 2118: Scientific Computing
Task 16 – Numerical Integration
Real-World Case: Estimating Total Distance Travelled by a Train

A train's speed v(t) is measured at regular intervals (every minute).
Total distance = ∫₀ᵀ v(t) dt

We approximate this integral using three methods:
  1. Trapezoidal Rule  (scipy.integrate.trapezoid)
  2. Simpson's Rule    (scipy.integrate.simpson)
  3. Gaussian Quadrature (scipy.integrate.quad) – for a known function

Then we compare the numerical results to the exact analytic integral.

Speed function (chosen for interesting shape and known antiderivative):
  v(t) = 60 + 20 * sin(π*t/30)   km/h, over 0 to 60 minutes

Exact integral (displacement):
  ∫₀⁶⁰ [60 + 20 sin(πt/30)] dt = 60*60 − 20*(30/π)*[cos(2π) − cos(0)] = 3600 km·min
  Convert to km: 3600/60 = 60 km
"""

import numpy as np
from scipy.integrate import trapezoid, simpson, quad
import matplotlib.pyplot as plt

# ── 1. Known speed function ───────────────────────────────────────────────────
def v(t):
    return 60 + 20 * np.sin(np.pi * t / 30)   # km/h

T = 60   # minutes
exact_distance_km = 60 + (20 * 30 / np.pi) * (np.cos(0) - np.cos(np.pi * T / 30))
# Using Wolfram: ∫₀⁶⁰ (60 + 20 sin(πt/30)) dt = [60t - (600/π)cos(πt/30)]₀⁶⁰
val = 60*T - (600/np.pi)*(np.cos(np.pi*T/30) - np.cos(0))
exact_km = val / 60.0   # convert from km·min/h to km (divide by 60)
print(f"=== Train Distance Integration ===")
print(f"  Exact distance (analytic): {exact_km:.6f} km")

# ── 2. Trapezoidal Rule ───────────────────────────────────────────────────────
n_points = 7   # 7 sample points
t_arr = np.linspace(0, T, n_points)
v_arr = v(t_arr)

trap_result_raw = trapezoid(v_arr, t_arr) / 60.0   # km·min → km
print(f"\n  Trapezoidal Rule  (n={n_points} points): {trap_result_raw:.6f} km  "
      f"error = {abs(trap_result_raw - exact_km):.6f}")

# More points = better accuracy
for n in [7, 11, 21, 61, 121]:
    t_n = np.linspace(0, T, n)
    v_n = v(t_n)
    trap_n = trapezoid(v_n, t_n) / 60.0
    print(f"    n={n:4d}: {trap_n:.8f} km  error={abs(trap_n - exact_km):.2e}")

# ── 3. Simpson's Rule ─────────────────────────────────────────────────────────
# Simpson requires an odd number of points (even number of intervals)
t_simp = np.linspace(0, T, 61)   # 61 points = 60 intervals (even)
v_simp = v(t_simp)
simp_result = simpson(v_simp, x=t_simp) / 60.0

print(f"\n  Simpson's Rule    (n=61 points): {simp_result:.8f} km  "
      f"error = {abs(simp_result - exact_km):.2e}")

# ── 4. Gaussian Quadrature via scipy.integrate.quad ──────────────────────────
# quad integrates the function directly (no need to sample first)
def v_km(t):
    return v(t) / 60.0   # convert to km/min so we integrate in minutes

gauss_result, gauss_error = quad(v_km, 0, T)
print(f"\n  Gaussian quad:    {gauss_result:.8f} km  estimated error = {gauss_error:.2e}")

# ── 5. Experiment – busier schedule (higher speed) ───────────────────────────
def v2(t):
    return 80 + 30 * np.sin(np.pi * t / 30)   # faster train

t_exp = np.linspace(0, T, 61)
result_exp = simpson(v2(t_exp), x=t_exp) / 60.0
print(f"\n=== Experiment: Faster train (v = 80 + 30 sin) ===")
print(f"  Total distance = {result_exp:.4f} km  (was {simp_result:.4f} km)")
print("  Higher average speed → greater distance covered in same time.")

# ── 6. Visualisation ─────────────────────────────────────────────────────────
t_fine = np.linspace(0, T, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Speed curve and integration area
axes[0].fill_between(t_fine, v(t_fine), alpha=0.2, color="steelblue", label="Area = distance")
axes[0].plot(t_fine, v(t_fine), color="steelblue", linewidth=2, label="v(t)")
axes[0].plot(t_arr, v_arr, "ro", markersize=5, label="Trapezoid sample points")
axes[0].set_title("Train Speed Curve  v(t) = 60 + 20sin(πt/30)")
axes[0].set_xlabel("Time (minutes)")
axes[0].set_ylabel("Speed (km/h)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Error vs number of intervals (trapezoidal)
ns = [3, 5, 7, 11, 21, 31, 61, 121, 241]
errors_trap = []
for n_i in ns:
    t_i = np.linspace(0, T, n_i)
    v_i = v(t_i)
    errors_trap.append(abs(trapezoid(v_i, t_i) / 60.0 - exact_km))

axes[1].loglog(ns, errors_trap, "o-", color="crimson", label="Trapezoidal error")
axes[1].set_title("Trapezoidal Error vs Number of Points")
axes[1].set_xlabel("n (number of sample points)")
axes[1].set_ylabel("|error| (km, log scale)")
axes[1].legend()
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_16_integration.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_16_integration.png")
