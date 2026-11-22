"""
BIT 2118: Scientific Computing
Individual Real-World Extensions – Tasks 02 to 20

Each section extends the original task to solve a different but related
real-world problem using the same mathematical concept.

Run the full file, or copy individual sections into separate scripts.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import solve_ivp, simpson
import matplotlib
matplotlib.use("Agg")     # save without displaying (safe for batch runs)
import matplotlib.pyplot as plt
import math
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 02 – NumPy Arrays: Weekly Sales Data Analysis (Retail Store)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 02: Retail Store Weekly Sales ─────────────────────────")

rng = np.random.default_rng(seed=11)
# 5 products × 7 days, sales in units
sales = rng.integers(10, 80, size=(5, 7)).astype(float)
products = ["Milk", "Bread", "Eggs", "Butter", "Juice"]
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

weekly_total = sales.sum(axis=1)
best_product = products[np.argmax(weekly_total)]
daily_avg    = sales.mean(axis=0)
best_day     = days[np.argmax(daily_avg)]

print(f"  Best product this week : {best_product} ({weekly_total.max():.0f} units)")
print(f"  Best sales day         : {best_day}")
print(f"  Products below 30 avg  : {[(products[i]) for i in range(5) if sales[i].mean() < 30]}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 03 – Vectorization: Batch Grade Conversion (% → Letter)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 03: Batch Grade Conversion ────────────────────────────")

rng2 = np.random.default_rng(seed=5)
scores = rng2.integers(30, 101, size=500).astype(float)   # 500 student scores

# Vectorized grade boundaries (no loop needed)
grades = np.select(
    [scores >= 70, scores >= 60, scores >= 50, scores >= 40],
    ["A",          "B",          "C",          "D"],
    default="F"
)

unique, counts = np.unique(grades, return_counts=True)
print("  Grade distribution:")
for g, c in sorted(zip(unique, counts)):
    print(f"    {g}: {c} students ({c/len(scores)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 04 – Floating Point: Interest Accumulation Error
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 04: Bank Interest Accumulation Error ──────────────────")

principal = 1000.0
monthly_rate = 0.005   # 0.5% per month
months = 120           # 10 years

# Float loop
balance_float = principal
for _ in range(months):
    balance_float += balance_float * monthly_rate

# Exact with Decimal
balance_dec = Decimal("1000")
rate_dec = Decimal("0.005")
for _ in range(months):
    balance_dec += balance_dec * rate_dec

exact = float(balance_dec)
print(f"  Float result  : ${balance_float:.6f}")
print(f"  Decimal result: ${exact:.6f}")
print(f"  Error         : ${abs(balance_float - exact):.8f}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 05 – Matrix Algebra: Student Mark Weightings
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 05: Student Final Mark Computation ────────────────────")

# 4 students × 3 assessments (Test1, Assignment, Exam)
marks = np.array([
    [65, 70, 72],
    [80, 75, 88],
    [55, 60, 50],
    [90, 95, 93]
], dtype=float)

# Weights: Test 20%, Assignment 30%, Exam 50%
weights = np.array([0.20, 0.30, 0.50])
finals = marks @ weights   # matrix-vector multiplication

print("  Student final marks:")
for i, f in enumerate(finals, start=1):
    print(f"    Student {i}: {f:.1f}%  → {'Pass' if f >= 50 else 'Fail'}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 06 – Linear Equations: Nutritional Meal Planning
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 06: Meal Plan – Calorie/Protein/Carb Targets ─────────")

# Daily targets: 2000 cal, 100g protein, 250g carbs
# Foods: Rice (per 100g), Chicken (per 100g), Broccoli (per 100g)
# Nutrients per 100g:
#              Calories  Protein  Carbs
A_meal = np.array([
    [130,   27,   28],   # rice, chicken, broccoli → calories
    [  3,   31,    3],   # → protein (g)
    [ 28,    0,    7],   # → carbs (g)
], dtype=float)

b_meal = np.array([2000, 100, 250], dtype=float)   # targets

# Solve: how many 100g portions of each food?
x_meal = np.linalg.solve(A_meal, b_meal)
foods = ["Rice (×100g)", "Chicken (×100g)", "Broccoli (×100g)"]
print("  Recommended portions:")
for food, qty in zip(foods, x_meal):
    print(f"    {food}: {qty:.2f} portions")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 07 – Gaussian/LU: Traffic Signal Timing Equations
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 07: Traffic Signal Timing ─────────────────────────────")

# 3 intersections; flow conservation gives 3 equations
A_traffic = np.array([
    [3,  1,  0],
    [1,  4,  2],
    [0,  2,  5]
], dtype=float)
b_traffic = np.array([120, 300, 250], dtype=float)   # vehicles/hr target

lu_fac = lu_factor(A_traffic)
t_signal = lu_solve(lu_fac, b_traffic)
print("  Green-light durations (seconds):")
for i, t in enumerate(t_signal, start=1):
    print(f"    Intersection {i}: {t:.2f} s")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 08 – Iterative Methods: Pressure in a Pipeline Network
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 08: Pipeline Pressure Distribution ────────────────────")

# 4-node pipeline; pressure at node 1 = 100 kPa, node 4 = 10 kPa
A_pipe = np.array([
    [-2,  1,  0,  0],
    [ 1, -2,  1,  0],
    [ 0,  1, -2,  1],
    [ 0,  0,  1, -2]
], dtype=float)
b_pipe = np.array([-100, 0, 0, -10], dtype=float)

x_pipe = np.linalg.solve(A_pipe, b_pipe)
print(f"  Pressures at interior nodes (kPa): {np.round(x_pipe, 2)}")
print("  Pressure decreases linearly along the pipe – consistent with laminar flow.")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 09 – Eigenvalues: Principal Component-like Analysis on Sensor Data
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 09: Sensor Covariance – Dominant Directions ───────────")

rng3 = np.random.default_rng(seed=77)
# Simulate 3 correlated sensors (temperature, humidity, pressure)
raw = rng3.normal(size=(3, 100))
# Introduce correlation
raw[1] = 0.8 * raw[0] + 0.2 * raw[1]
raw[2] = 0.5 * raw[0] + 0.5 * raw[2]

cov_matrix = np.cov(raw)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx].real

total_var = eigenvalues.sum()
print("  Eigenvalues (variance explained):")
for i, lam in enumerate(eigenvalues):
    print(f"    Component {i+1}: {lam:.4f}  ({lam/total_var*100:.1f}% of variance)")
print("  First component dominates → sensors are highly correlated.")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 10 – Bisection: Finding Bond Yield (Finance)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 10: Bond Yield Calculation (Bisection) ────────────────")

# A bond: face value = $1000, coupon = $50/year, maturity = 10 years
# Current price = $950. Find yield (y) satisfying:
# 950 = sum_{t=1}^{10} [50 / (1+y)^t] + 1000 / (1+y)^10

def bond_price(y):
    coupon = 50
    face   = 1000
    n      = 10
    pv_coupons = sum(coupon / (1 + y)**t for t in range(1, n + 1))
    pv_face    = face / (1 + y)**n
    return pv_coupons + pv_face - 950   # equation = 0 when y is correct

yield_root = brentq(bond_price, 0.001, 0.5)
print(f"  Bond yield (YTM): {yield_root*100:.4f}%")
print(f"  Verification: bond_price_eq({yield_root:.6f}) = {bond_price(yield_root):.6f}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 11 – Newton-Raphson: Compound Interest Rate Solver
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 11: Finding Interest Rate (Newton-Raphson) ────────────")

# Invest $5000, want $8000 in 6 years.  5000*(1+r)^6 = 8000
# f(r) = 5000*(1+r)^6 - 8000 = 0
def f_interest(r):
    return 5000 * (1 + r)**6 - 8000

def df_interest(r):
    return 5000 * 6 * (1 + r)**5

r = 0.1   # initial guess
for i in range(20):
    r_new = r - f_interest(r) / df_interest(r)
    if abs(r_new - r) < 1e-10:
        break
    r = r_new

print(f"  Required annual interest rate: {r_new*100:.4f}%")
print(f"  Verification: 5000*(1+r)^6 = {5000*(1+r_new)**6:.2f}  (target 8000)")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 12 – Polynomial/Nonlinear: Fuel Consumption vs Speed
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 12: Optimal Driving Speed for Fuel Efficiency ─────────")

# Fuel consumption model: C(v) = 0.002*v^2 - 0.3*v + 15  (litres/100km)
def fuel(v):
    return 0.002 * v**2 - 0.3 * v + 15

res_fuel = minimize_scalar(fuel, bounds=(40, 150), method="bounded")
print(f"  Most fuel-efficient speed: {res_fuel.x:.2f} km/h")
print(f"  Fuel consumption at that speed: {res_fuel.fun:.4f} L/100km")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 13 – Interpolation: Stock Price Estimation
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 13: Stock Price Interpolation ─────────────────────────")

trading_days = np.array([1, 3, 5, 8, 10, 12, 15], dtype=float)
prices       = np.array([145.0, 148.5, 143.2, 150.1, 155.0, 149.8, 160.3], dtype=float)

cs_stock = CubicSpline(trading_days, prices)
query_days = np.array([2, 4, 7, 11, 14], dtype=float)
print("  Estimated stock prices on missing trading days:")
for d, p in zip(query_days, cs_stock(query_days)):
    print(f"    Day {d:2.0f}: ${p:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 14 – Curve Fitting: Internet Usage Growth Over Years
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 14: Internet Usage Growth (Exponential Fit) ───────────")

years_offset = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)   # years since 2017
users_m = np.array([3.8, 4.0, 4.2, 4.5, 4.9, 5.1, 5.3, 5.5], dtype=float)   # billions

coeffs_inet = np.polyfit(years_offset, users_m, deg=1)
p_inet = np.poly1d(coeffs_inet)
print(f"  Linear fit: users = {coeffs_inet[0]:.4f}*year + {coeffs_inet[1]:.4f} billion")
pred_2027 = p_inet(10)
print(f"  Predicted users in 2027 (year offset 10): {pred_2027:.2f} billion")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 15 – Differentiation: Marginal Cost in Economics
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 15: Marginal Cost via Numerical Differentiation ────────")

# Total cost function: TC(q) = q^3 - 6q^2 + 15q + 10
def TC(q):
    return q**3 - 6*q**2 + 15*q + 10

# Marginal cost = dTC/dq (central difference)
q_values = np.arange(1, 11, dtype=float)
h = 0.001
MC_num   = (TC(q_values + h) - TC(q_values - h)) / (2 * h)
MC_exact = 3*q_values**2 - 12*q_values + 15

print(f"  {'q':>4}  {'MC (numerical)':>16}  {'MC (exact)':>12}")
for q, mc_n, mc_e in zip(q_values, MC_num, MC_exact):
    print(f"  {q:>4.0f}  {mc_n:>16.4f}  {mc_e:>12.4f}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 16 – Integration: Work Done by a Variable Force
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 16: Work Done by Variable Force ────────────────────────")

# F(x) = 5 + 3*sin(x)  Newtons,  x from 0 to π
x_force = np.linspace(0, np.pi, 101)
F_vals  = 5 + 3 * np.sin(x_force)
work    = simpson(F_vals, x=x_force)
exact_work = 5*np.pi + 3*(-np.cos(np.pi) + np.cos(0))   # = 5π + 6

print(f"  Numerical work (Simpson): {work:.6f} J")
print(f"  Exact work              : {exact_work:.6f} J")
print(f"  Error                   : {abs(work - exact_work):.2e} J")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 17 – Euler ODE: Drug Concentration in Bloodstream
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 17: Drug Concentration (Euler Method) ─────────────────")

# dC/dt = -k*C  (first-order clearance)  k = 0.3 /hr
# Dose administered: C0 = 200 mg/L at t=0
k_drug = 0.3
C0_drug = 200.0

def f_drug(t, C):
    return -k_drug * C

h_drug = 0.5   # hours
t_drug = np.arange(0, 12 + h_drug, h_drug)
C_euler = [C0_drug]
for i in range(len(t_drug) - 1):
    C_new = C_euler[-1] + h_drug * f_drug(t_drug[i], C_euler[-1])
    C_euler.append(C_new)
C_euler = np.array(C_euler)
C_exact_drug = C0_drug * np.exp(-k_drug * t_drug)

print(f"  {'t (hr)':>8}  {'Euler C':>10}  {'Exact C':>10}")
for i in range(0, len(t_drug), 4):
    print(f"  {t_drug[i]:>8.1f}  {C_euler[i]:>10.4f}  {C_exact_drug[i]:>10.4f}")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 18 – SciPy ODE: SIR Epidemic Model
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 18: SIR Epidemic Model (solve_ivp) ────────────────────")

# S: susceptible, I: infected, R: recovered
# dS/dt = -beta*S*I,  dI/dt = beta*S*I - gamma*I,  dR/dt = gamma*I
beta_sir  = 0.3    # transmission rate
gamma_sir = 0.05   # recovery rate
N_pop     = 1000.0

def sir(t, y):
    S, I, R = y
    dS = -beta_sir * S * I / N_pop
    dI =  beta_sir * S * I / N_pop - gamma_sir * I
    dR =  gamma_sir * I
    return [dS, dI, dR]

y0_sir = [990, 10, 0]
sol_sir = solve_ivp(sir, (0, 160), y0_sir, method="RK45",
                    t_eval=np.linspace(0, 160, 500), rtol=1e-6, atol=1e-8)

peak_infected = sol_sir.y[1].max()
t_peak = sol_sir.t[np.argmax(sol_sir.y[1])]
print(f"  Peak infections: {peak_infected:.0f} people at t = {t_peak:.1f} days")
print(f"  Total recovered: {sol_sir.y[2][-1]:.0f} people")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 19 – Optimization: Minimum-Cost Box Design
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 19: Optimal Box Dimensions (Min Cost) ─────────────────")

# Open-top rectangular box, volume = 500 cm³
# Cost: base costs $2/cm², sides cost $1/cm²
# Volume: l*w*h = 500, l = w (square base for simplicity)
# Surface area cost = 2*l^2 + 4*l*h, h = 500/l^2
def box_cost(l):
    h = 500 / l**2
    base_area  = l**2
    side_area  = 4 * l * h
    return 2 * base_area + 1 * side_area

res_box = minimize_scalar(box_cost, bounds=(1, 50), method="bounded")
l_opt = res_box.x
h_opt = 500 / l_opt**2
print(f"  Optimal base length : {l_opt:.4f} cm")
print(f"  Optimal height      : {h_opt:.4f} cm")
print(f"  Minimum cost        : ${res_box.fun:.4f}")
print(f"  Volume check        : {l_opt**2 * h_opt:.2f} cm³  (target 500)")

# ═══════════════════════════════════════════════════════════════════════
# EXTENSION 20 – Monte Carlo: Estimating Probability of Password Strength
# ═══════════════════════════════════════════════════════════════════════
print("\n── Extension 20: Monte Carlo – Network Packet Loss Simulation ───────")

# Simulate a network with 3 links in series.
# Each link independently fails with probability p_fail.
# We want P(at least one link fails) over 100 000 trials.
p_fail = 0.05    # 5% chance each link fails
N_net = 100_000
rng4 = np.random.default_rng(seed=200)

link1 = rng4.uniform(0, 1, N_net) < p_fail
link2 = rng4.uniform(0, 1, N_net) < p_fail
link3 = rng4.uniform(0, 1, N_net) < p_fail

connection_failed = link1 | link2 | link3   # fails if ANY link fails
p_connection_fail = connection_failed.mean()
p_exact = 1 - (1 - p_fail)**3

print(f"  Trials: {N_net:,}")
print(f"  P(connection fails) – Monte Carlo: {p_connection_fail:.4f}")
print(f"  P(connection fails) – Exact      : {p_exact:.4f}")
print(f"  Difference: {abs(p_connection_fail - p_exact):.4f}")

print("\n── All individual extensions completed ─────────────────────────────")
