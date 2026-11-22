"""
BIT 2118: Scientific Computing
Task 13 – Interpolation
Real-World Case: Estimating Missing Air Temperature Readings

A weather station records hourly temperatures but some readings are missing.
We have recorded temperatures at 7 specific hours (not evenly spaced).
We use interpolation to estimate the temperature at any hour in between.

Three methods compared:
  1. Linear interpolation (numpy)
  2. Polynomial interpolation (numpy.polyfit / numpy.polyval)
  3. Cubic spline interpolation (scipy.interpolate.CubicSpline)

Data:
  Hours:       0,  3,  6,  9, 12, 18, 24
  Temperature: 15, 13, 11, 16, 24, 22, 17  (°C)
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ── 1. Known data points ──────────────────────────────────────────────────────
hours = np.array([ 0,  3,  6,  9, 12, 18, 24], dtype=float)
temps = np.array([15, 13, 11, 16, 24, 22, 17], dtype=float)

# Hours where we want to estimate temperature
query_hours = np.array([2, 5, 10, 15, 20], dtype=float)

print("=== Temperature Interpolation ===")
print(f"  Known data points: {list(zip(hours.astype(int), temps))}")

# ── 2. Linear Interpolation ───────────────────────────────────────────────────
temps_linear = np.interp(query_hours, hours, temps)
print(f"\n  Linear interpolation at hours {list(query_hours.astype(int))}:")
for h, t in zip(query_hours, temps_linear):
    print(f"    Hour {h:2.0f}: {t:.2f} °C")

# ── 3. Polynomial Interpolation (degree 6 to pass through all 7 points) ───────
coeffs = np.polyfit(hours, temps, deg=6)
poly_func = np.poly1d(coeffs)
temps_poly = poly_func(query_hours)
print(f"\n  Polynomial (degree 6) at hours {list(query_hours.astype(int))}:")
for h, t in zip(query_hours, temps_poly):
    print(f"    Hour {h:2.0f}: {t:.2f} °C")

# ── 4. Cubic Spline Interpolation ─────────────────────────────────────────────
cs = CubicSpline(hours, temps)
temps_spline = cs(query_hours)
print(f"\n  Cubic spline at hours {list(query_hours.astype(int))}:")
for h, t in zip(query_hours, temps_spline):
    print(f"    Hour {h:2.0f}: {t:.2f} °C")

# ── 5. Experiment – change data point at hour 12 (cloudy day) ────────────────
temps_cloudy = np.array([15, 13, 11, 14, 17, 15, 14], dtype=float)   # cooler day
cs_cloudy = CubicSpline(hours, temps_cloudy)
print(f"\n=== Experiment: Cooler (cloudy) day temperatures ===")
for h in query_hours:
    print(f"    Hour {h:2.0f}: {cs_cloudy(h):.2f} °C")
print("  Lower peak temperature → spline curve is flatter throughout the day.")

# ── 6. Visualisation ─────────────────────────────────────────────────────────
h_fine = np.linspace(0, 24, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Original data comparison
axes[0].plot(hours,  temps, "ko", markersize=7, label="Measured data", zorder=5)
axes[0].plot(h_fine, np.interp(h_fine, hours, temps), "-",  color="blue",   label="Linear")
axes[0].plot(h_fine, poly_func(h_fine),                 "--", color="orange", label="Polynomial (deg 6)")
axes[0].plot(h_fine, cs(h_fine),                        "-.", color="green",  label="Cubic Spline")
axes[0].set_title("Temperature Interpolation Methods")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Experiment comparison
axes[1].plot(hours, temps,         "ko",   label="Sunny day data")
axes[1].plot(hours, temps_cloudy,  "k^",   label="Cloudy day data")
axes[1].plot(h_fine, cs(h_fine),           "-",  color="orange", label="Spline – sunny")
axes[1].plot(h_fine, cs_cloudy(h_fine),    "--", color="steelblue", label="Spline – cloudy")
axes[1].set_title("Experiment: Sunny vs Cloudy Day")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_13_interpolation.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_13_interpolation.png")
