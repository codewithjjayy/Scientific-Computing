"""
Task 12 – Polynomial and Nonlinear Equation Solving
Real-World Case: Projectile Range – Finding Launch Angles

A ball is launched at speed v0 = 20 m/s. The horizontal range is:
  R(θ) = (v0² * sin(2θ)) / g

We want to find the launch angle θ (in degrees) that gives a target
range of R_target = 35 metres. This becomes:

  f(θ) = (v0² * sin(2θ)) / g − 35 = 0

We solve using:
  1. numpy.roots for a polynomial example (separate demo)
  2. scipy.optimize.brentq for the nonlinear equation
  3. scipy.optimize.fsolve for a general nonlinear system
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt

#  1. Polynomial root finding using numpy.roots 
# Example polynomial: p(x) = x³ - 6x² + 11x - 6 = 0
# Known roots: x = 1, 2, 3
coeffs = [1, -6, 11, -6]   # coefficients from highest to lowest degree
roots_poly = np.roots(coeffs)

print("=== Polynomial  x³ - 6x² + 11x - 6 ===")
print(f"  Roots (numpy.roots): {np.round(roots_poly, 6)}")
print(f"  Expected: [1, 2, 3]")

#  2. Nonlinear equation – projectile range 
v0       = 20.0   # m/s
g        = 9.81   # m/s²
R_target = 35.0   # m

def range_eq(theta_deg):
    theta_rad = np.radians(theta_deg)
    return (v0**2 * np.sin(2 * theta_rad)) / g - R_target

print(f"\n=== Projectile Range  v0={v0} m/s, R_target={R_target} m ===")

# Two solutions exist between 0° and 90° for most targets (< max range)
max_range = v0**2 / g
print(f"  Maximum possible range = {max_range:.2f} m")

# Solution 1: low angle (< 45°)
theta1 = brentq(range_eq, 1, 44)
# Solution 2: high angle (> 45°)
theta2 = brentq(range_eq, 46, 89)

print(f"  Low-angle solution  : θ = {theta1:.4f}°")
print(f"  High-angle solution : θ = {theta2:.4f}°")
print(f"  Verification: R(θ1) = {((v0**2 * np.sin(2*np.radians(theta1)))/g):.4f} m")
print(f"  Verification: R(θ2) = {((v0**2 * np.sin(2*np.radians(theta2)))/g):.4f} m")

# ── 3. fsolve 
general nonlinear system: sin(x) - y = 0  and  x² + y² - 1 = 0  (circle-sine intersection)
def system(vars):
    x, y = vars
    eq1 = np.sin(x) - y
    eq2 = x**2 + y**2 - 1
    return [eq1, eq2]

sol = fsolve(system, x0=[0.5, 0.5], full_output=True)
x_sol, y_sol = sol[0]
print(f"\n=== fsolve: sin(x)=y and x²+y²=1 ===")
print(f"  Solution: x = {x_sol:.6f}, y = {y_sol:.6f}")
print(f"  sin(x) = {np.sin(x_sol):.6f}, y = {y_sol:.6f} ✓")

# 4. Experiment – change target range
R_new = 25.0
def range_eq2(theta_deg):
    return (v0**2 * np.sin(2 * np.radians(theta_deg))) / g - R_new

theta1b = brentq(range_eq2, 1, 44)
theta2b = brentq(range_eq2, 46, 89)
print(f"\n=== Experiment: R_target = {R_new} m ===")
print(f"  Low-angle : θ = {theta1b:.4f}°  (was {theta1:.4f}°)")
print(f"  High-angle: θ = {theta2b:.4f}°  (was {theta2:.4f}°)")
print("  Shorter range → flatter and steeper angles that are closer to 45°.")

# 5. Visualisation 
theta_arr = np.linspace(1, 89, 300)
R_arr = (v0**2 * np.sin(2 * np.radians(theta_arr))) / g

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(theta_arr, R_arr, color="steelblue", label="Range R(θ)")
axes[0].axhline(R_target, color="red",   linestyle="--", label=f"Target {R_target} m")
axes[0].axvline(theta1,   color="green", linestyle=":",  label=f"θ₁ = {theta1:.1f}°")
axes[0].axvline(theta2,   color="orange",linestyle=":",  label=f"θ₂ = {theta2:.1f}°")
axes[0].set_title("Projectile Range vs Launch Angle")
axes[0].set_xlabel("Launch angle θ (°)")
axes[0].set_ylabel("Horizontal range R (m)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Polynomial: plot p(x) and mark roots
x_p = np.linspace(0, 4, 300)
y_p = x_p**3 - 6*x_p**2 + 11*x_p - 6
axes[1].plot(x_p, y_p, color="purple", label="p(x) = x³−6x²+11x−6")
axes[1].axhline(0, color="black", linewidth=0.8)
for r in [1, 2, 3]:
    axes[1].axvline(r, color="crimson", linestyle="--", alpha=0.6)
    axes[1].text(r + 0.05, 0.3, f"x={r}", color="crimson", fontsize=9)
axes[1].set_title("Polynomial Root Finding")
axes[1].set_xlabel("x")
axes[1].set_ylabel("p(x)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_12_polynomial_nonlinear.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_12_polynomial_nonlinear.png")
