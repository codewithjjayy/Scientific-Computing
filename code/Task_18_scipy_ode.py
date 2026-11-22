"""
Task 18 – Numerical Solution of ODEs Using SciPy
Real-World Case: Predator-Prey Population Dynamics (Lotka-Volterra Model)

The Lotka-Volterra equations model how two species interact:
  - Prey (rabbits): grow without predators; eaten by predators
  - Predators (foxes): die without prey; grow by eating prey

Equations:
  dR/dt = alpha * R  -  beta  * R * F     (rabbit population)
  dF/dt = delta * R * F  -  gamma * F     (fox population)

Parameters:
  alpha = 0.1   (rabbit birth rate)
  beta  = 0.02  (predation rate)
  delta = 0.01  (fox growth from eating rabbits)
  gamma = 0.1   (fox death rate)

Initial conditions: R0 = 40 rabbits, F0 = 9 foxes

We use scipy.integrate.solve_ivp with the RK45 method.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define the ODE system
alpha = 0.1    # rabbit birth rate
beta  = 0.02   # predation rate
delta = 0.01   # fox growth per rabbit eaten
gamma = 0.1    # fox death rate

def lotka_volterra(t, y):
    R, F = y
    dR_dt = alpha * R - beta  * R * F
    dF_dt = delta * R * F - gamma * F
    return [dR_dt, dF_dt]

#  2. Solve using solve_ivp (RK45) 
t_span = (0, 200)
t_eval = np.linspace(0, 200, 2000)
y0 = [40, 9]   # initial: 40 rabbits, 9 foxes

sol = solve_ivp(lotka_volterra, t_span, y0, method="RK45", t_eval=t_eval,
                rtol=1e-6, atol=1e-8)

R = sol.y[0]
F = sol.y[1]
t = sol.t

print("=== Lotka-Volterra Predator-Prey Model ===")
print(f"  Parameters: α={alpha}, β={beta}, δ={delta}, γ={gamma}")
print(f"  Initial:    R0={y0[0]} rabbits,  F0={y0[1]} foxes")
print(f"  Duration:   {t_span[1]} time units")
print(f"\n  Max rabbit population : {R.max():.1f}")
print(f"  Max fox population    : {F.max():.1f}")
print(f"  Min rabbit population : {R.min():.1f}")
print(f"  Min fox population    : {F.min():.1f}")

#  3. Experiment – higher predation rate 
beta2 = 0.04
def lv2(t, y):
    R, F = y
    return [alpha*R - beta2*R*F,  delta*R*F - gamma*F]

sol2 = solve_ivp(lv2, t_span, y0, method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8)
R2, F2 = sol2.y[0], sol2.y[1]
print(f"\n=== Experiment: beta doubled to {beta2} (higher predation) ===")
print(f"  Max rabbit: {R2.max():.1f}  (was {R.max():.1f})")
print(f"  Max fox   : {F2.max():.1f}  (was {F.max():.1f})")
print("  Higher predation rate → rabbit population drops faster → fox peaks differently.")

# 4. Visualisation 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Time series
axes[0].plot(t, R, color="steelblue", label="Rabbits (prey)")
axes[0].plot(t, F, color="crimson",   label="Foxes (predator)")
axes[0].set_title("Lotka-Volterra Population Dynamics")
axes[0].set_xlabel("Time (arbitrary units)")
axes[0].set_ylabel("Population size")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Phase portrait (R vs F)
axes[1].plot(R, F, color="purple", linewidth=0.9, label="Phase trajectory")
axes[1].plot(y0[0], y0[1], "go", markersize=8, label="Start")
axes[1].set_title("Phase Portrait: Rabbits vs Foxes")
axes[1].set_xlabel("Rabbit population")
axes[1].set_ylabel("Fox population")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_18_scipy_ode.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_18_scipy_ode.png")
