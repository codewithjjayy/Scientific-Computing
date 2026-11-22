"""
BIT 2118: Scientific Computing
Task 17 – Ordinary Differential Equations: Euler Method
Real-World Case: Cooling of a Hot Cup of Coffee (Newton's Law of Cooling)

Newton's Law of Cooling says:
  dT/dt = -k * (T - T_ambient)

where:
  T         = temperature of the coffee at time t (°C)
  T_ambient = room temperature = 22°C
  k         = cooling constant = 0.07 per minute
  T0        = initial temperature = 95°C

The exact (analytic) solution is:
  T(t) = T_ambient + (T0 - T_ambient) * e^(-k*t)

Euler's method approximates the solution step by step:
  T_{n+1} = T_n + h * f(t_n, T_n)
  where f(t, T) = -k * (T - T_ambient)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Constants ──────────────────────────────────────────────────────────────
T_ambient = 22.0    # °C (room temperature)
k         = 0.07    # cooling rate (1/min)
T0        = 95.0    # °C (initial coffee temperature)
t_end     = 60.0    # simulate for 60 minutes

# ODE right-hand side
def f(t, T):
    return -k * (T - T_ambient)

# Exact (analytic) solution
def T_exact(t):
    return T_ambient + (T0 - T_ambient) * np.exp(-k * t)

# ── 2. Euler's Method ─────────────────────────────────────────────────────────
def euler(f, T0, t0, t_end, h):
    t_vals = [t0]
    T_vals = [T0]
    t = t0
    T = T0
    while t < t_end - 1e-10:
        T = T + h * f(t, T)
        t = t + h
        t_vals.append(t)
        T_vals.append(T)
    return np.array(t_vals), np.array(T_vals)

# Large step (h = 5 min) – less accurate
t5, T5 = euler(f, T0, 0, t_end, h=5)
# Smaller step (h = 1 min) – more accurate
t1, T1 = euler(f, T0, 0, t_end, h=1)
# Finer step (h = 0.5 min) – most accurate
t05, T05 = euler(f, T0, 0, t_end, h=0.5)

t_fine = np.linspace(0, t_end, 300)
T_true = T_exact(t_fine)

print("=== Euler Method – Coffee Cooling ===")
print(f"  T_ambient = {T_ambient}°C,  k = {k},  T0 = {T0}°C")
print(f"\n  t (min)  |  Euler h=1  |  Exact  |  Error")
for i in range(0, len(t1), 10):
    t_i = t1[i]
    err = abs(T1[i] - T_exact(t_i))
    print(f"  {t_i:7.1f}  |  {T1[i]:9.4f}  |  {T_exact(t_i):7.4f}  |  {err:.4f}")

print(f"\n  Final temperature at t={t_end:.0f} min:")
print(f"    Euler h=5  : {T5[-1]:.4f} °C")
print(f"    Euler h=1  : {T1[-1]:.4f} °C")
print(f"    Euler h=0.5: {T05[-1]:.4f} °C")
print(f"    Exact      : {T_exact(t_end):.4f} °C")
print("  → Smaller h → closer to exact solution.")

# ── 3. Experiment – higher cooling constant (k = 0.15, e.g. iced drink) ──────
k2 = 0.15
def f2(t, T):
    return -k2 * (T - T_ambient)
def T_exact2(t):
    return T_ambient + (T0 - T_ambient) * np.exp(-k2 * t)

t1b, T1b = euler(f2, T0, 0, t_end, h=1)
print(f"\n=== Experiment: k doubled to {k2} (faster cooling) ===")
print(f"  Temperature at 30 min: {T1b[30]:.2f} °C  (original: {T1[30]:.2f} °C)")
print("  Higher k → coffee cools much faster.")

# ── 4. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Temperature vs time for different step sizes
axes[0].plot(t_fine, T_true,  "-",  color="black",     linewidth=2, label="Exact solution")
axes[0].plot(t5,     T5,      "o--", color="red",      markersize=4, label="Euler h=5")
axes[0].plot(t1,     T1,      "s-",  color="orange",   markersize=3, label="Euler h=1")
axes[0].plot(t05,    T05,     "-.",  color="green",    linewidth=1.2, label="Euler h=0.5")
axes[0].axhline(T_ambient, color="gray", linestyle=":", label=f"Room temp {T_ambient}°C")
axes[0].set_title("Newton's Law of Cooling – Euler Method")
axes[0].set_xlabel("Time (minutes)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Error comparison vs time for h=1 and h=5
err5 = np.abs(T5 - T_exact(t5))
err1 = np.abs(T1 - T_exact(t1))
axes[1].plot(t5, err5, "o-",  color="red",    label="Error (h=5)")
axes[1].plot(t1, err1, "s--", color="orange", label="Error (h=1)")
axes[1].set_title("Euler Error vs Time")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_ylabel("Absolute error (°C)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_17_euler_ode.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_17_euler_ode.png")
