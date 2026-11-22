"""

Task 15 – Numerical Differentiation
Real-World Case: Estimating Instantaneous Speed of a Moving Vehicle

A GPS tracker records a car's position s(t) in metres at every second.
We need the velocity (ds/dt) and acceleration (d²s/dt²) at each instant
without knowing the exact function – only the data values.

Three finite-difference formulas:
  Forward difference  : f'(x) ≈ [f(x+h) - f(x)] / h
  Backward difference : f'(x) ≈ [f(x) - f(x-h)] / h
  Central difference  : f'(x) ≈ [f(x+h) - f(x-h)] / (2h)   ← most accurate

We also compare numerical derivatives against the exact analytic derivative
for a known function: s(t) = 3t³ - 2t² + t + 5

v(t) = ds/dt = 9t² - 4t + 1   (exact formula for verification)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Known function and exact derivative 
def s(t):
    return 3 * t**3 - 2 * t**2 + t + 5

def v_exact(t):
    return 9 * t**2 - 4 * t + 1

def a_exact(t):
    return 18 * t - 4

#  2. Numerical derivatives 
def forward_diff(func, t, h):
    return (func(t + h) - func(t)) / h

def backward_diff(func, t, h):
    return (func(t) - func(t - h)) / h

def central_diff(func, t, h):
    return (func(t + h) - func(t - h)) / (2 * h)

# 3. Compare at t = 2.0 s with h = 0.1 
t_pt = 2.0
h    = 0.1

v_fwd  = forward_diff(s, t_pt, h)
v_bwd  = backward_diff(s, t_pt, h)
v_cen  = central_diff(s, t_pt, h)
v_true = v_exact(t_pt)

print("=== Numerical Differentiation at t = 2.0 s,  h = 0.1 ===")
print(f"  Exact v(2)        : {v_true:.6f} m/s")
print(f"  Forward diff      : {v_fwd:.6f}  error = {abs(v_fwd - v_true):.6f}")
print(f"  Backward diff     : {v_bwd:.6f}  error = {abs(v_bwd - v_true):.6f}")
print(f"  Central diff      : {v_cen:.6f}  error = {abs(v_cen - v_true):.6f}")
print("  → Central difference has the smallest error.")

#  4. Error vs step size h
h_values = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
err_fwd, err_cen = [], []
for h_i in h_values:
    err_fwd.append(abs(forward_diff(s, t_pt, h_i) - v_true))
    err_cen.append(abs(central_diff(s, t_pt, h_i) - v_true))

print(f"\n  h        Forward error   Central error")
for h_i, ef, ec in zip(h_values, err_fwd, err_cen):
    print(f"  {h_i:<8}  {ef:.2e}       {ec:.2e}")

# 5. Apply to GPS data
# Simulated GPS: position recorded every second for 10 seconds
t_data = np.arange(0, 11, dtype=float)                     # 0 to 10 s
s_data = 3 * t_data**3 - 2 * t_data**2 + t_data + 5       # s(t)

# Central difference for interior points
h_data = 1.0
velocity_num = np.gradient(s_data, h_data)                 # numpy handles edges automatically
accel_num    = np.gradient(velocity_num, h_data)

print(f"\n=== GPS Position, Velocity, Acceleration (every 1 s) ===")
print(f"  {'t':>4}  {'s(m)':>8}  {'v_num':>8}  {'v_exact':>8}  {'a_num':>8}")
for i in range(len(t_data)):
    print(f"  {t_data[i]:>4.0f}  {s_data[i]:>8.2f}  {velocity_num[i]:>8.4f}  "
          f"{v_exact(t_data[i]):>8.4f}  {accel_num[i]:>8.4f}")

#  6. Experiment 
t_coarse = np.arange(0, 11, 2, dtype=float)
s_coarse = s(t_coarse)
v_coarse = np.gradient(s_coarse, 2.0)
print(f"\n=== Experiment: GPS sampled every 2 s (coarser) ===")
for i in range(len(t_coarse)):
    print(f"  t={t_coarse[i]:.0f}: v_num={v_coarse[i]:.4f}  v_exact={v_exact(t_coarse[i]):.4f}  "
          f"error={abs(v_coarse[i]-v_exact(t_coarse[i])):.4f}")
print("  Larger step → larger numerical error in estimated velocity.")

# 7. Visualisation 
t_fine = np.linspace(0, 10, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Position, velocity, acceleration
axes[0].plot(t_fine, s(t_fine),       "-",  color="steelblue", label="Position s(t)")
axes[0].plot(t_fine, v_exact(t_fine), "--", color="crimson",   label="Velocity v(t) [exact]")
axes[0].plot(t_data, velocity_num,    "o",  color="orange",    label="Velocity [numerical]", markersize=5)
axes[0].set_title("Position and Velocity of Car")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Error vs h
axes[1].loglog(h_values, err_fwd, "o-",  color="orange", label="Forward diff error")
axes[1].loglog(h_values, err_cen, "s--", color="green",  label="Central diff error")
axes[1].set_title("Error vs Step Size h")
axes[1].set_xlabel("h (step size)")
axes[1].set_ylabel("|error| (log scale)")
axes[1].legend()
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_15_differentiation.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_15_differentiation.png")
