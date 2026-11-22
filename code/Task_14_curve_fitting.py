"""
BIT 2118: Scientific Computing
Task 14 – Curve Fitting and Approximation
Real-World Case: Modelling a Smartphone Battery Discharge Curve

A phone is charged to 100% and discharged over 10 hours. The battery
percentage at selected time points is recorded.

We fit two models:
  1. Linear fit  – y = m*t + c
  2. Quadratic fit – y = a*t² + b*t + c

Using numpy.polyfit (least squares) and scipy.optimize.curve_fit.

Data collected every hour:
  t (hours): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
  Battery %: 100, 92, 83, 75, 65, 57, 46, 36, 25, 13, 3
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#  1. Recorded data 
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
battery = np.array([100, 92, 83, 75, 65, 57, 46, 36, 25, 13, 3], dtype=float)

print("=== Battery Discharge Curve Fitting ===")

# 2. Linear fit (degree 1) 
coeffs1 = np.polyfit(t, battery, deg=1)
p1 = np.poly1d(coeffs1)
residuals1 = battery - p1(t)
r2_linear = 1 - np.sum(residuals1**2) / np.sum((battery - battery.mean())**2)

print(f"\n  Linear fit:     y = {coeffs1[0]:.3f}*t + {coeffs1[1]:.3f}")
print(f"  R² (linear)   = {r2_linear:.4f}")

# 3. Quadratic fit (degree 2) 
coeffs2 = np.polyfit(t, battery, deg=2)
p2 = np.poly1d(coeffs2)
residuals2 = battery - p2(t)
r2_quad = 1 - np.sum(residuals2**2) / np.sum((battery - battery.mean())**2)

print(f"\n  Quadratic fit:  y = {coeffs2[0]:.4f}*t² + {coeffs2[1]:.3f}*t + {coeffs2[2]:.3f}")
print(f"  R² (quadratic)= {r2_quad:.4f}")

# 4. Exponential fit using curve_fit 
def exp_model(t, a, b):
    return a * np.exp(b * t)

popt, _ = curve_fit(exp_model, t, battery, p0=[100, -0.3])
a_fit, b_fit = popt
battery_exp = exp_model(t, a_fit, b_fit)
residuals3 = battery - battery_exp
r2_exp = 1 - np.sum(residuals3**2) / np.sum((battery - battery.mean())**2)

print(f"\n  Exponential fit: y = {a_fit:.3f} * e^({b_fit:.4f}*t)")
print(f"  R² (exp)       = {r2_exp:.4f}")

# Predict when battery reaches 20%
t_pred_lin  = (20 - coeffs1[1]) / coeffs1[0]
t_fine = np.linspace(0, 12, 300)
# Find crossing for quadratic
zeros_q = np.roots(coeffs2 - np.array([0, 0, 20]))
t_pred_quad = [z.real for z in zeros_q if np.isreal(z) and z.real > 0 and z.real < 12]

print(f"\n  Predicted 20% battery:")
print(f"    Linear    : t = {t_pred_lin:.2f} h")
if t_pred_quad:
    print(f"    Quadratic : t = {t_pred_quad[0]:.2f} h")

#  5. Experiment – faster discharge (heavy usage)
battery_fast = np.array([100, 85, 70, 54, 40, 28, 18, 10, 5, 2, 1], dtype=float)
coeffs_fast = np.polyfit(t, battery_fast, deg=2)
p_fast = np.poly1d(coeffs_fast)
print(f"\n=== Experiment: Heavy usage discharge ===")
print(f"  Quadratic fit: y = {coeffs_fast[0]:.4f}*t² + {coeffs_fast[1]:.3f}*t + {coeffs_fast[2]:.3f}")
print("  Battery drains much faster – the fitted curve falls more steeply.")

# 6. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

t_fine = np.linspace(0, 10, 300)
axes[0].scatter(t, battery, color="black", zorder=5, label="Measured data")
axes[0].plot(t_fine, p1(t_fine),                   "-",  color="blue",   label=f"Linear (R²={r2_linear:.3f})")
axes[0].plot(t_fine, p2(t_fine),                   "--", color="crimson",label=f"Quadratic (R²={r2_quad:.3f})")
axes[0].plot(t_fine, exp_model(t_fine, a_fit, b_fit), "-.", color="green", label=f"Exponential (R²={r2_exp:.3f})")
axes[0].set_title("Battery Discharge – Curve Fitting")
axes[0].set_xlabel("Time (hours)")
axes[0].set_ylabel("Battery (%)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Residuals plot for quadratic
axes[1].bar(t, residuals2, color=["green" if r >= 0 else "red" for r in residuals2])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Quadratic Fit Residuals (data − fit)")
axes[1].set_xlabel("Time (hours)")
axes[1].set_ylabel("Residual (%)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_14_curve_fitting.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_14_curve_fitting.png")
