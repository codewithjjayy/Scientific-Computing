"""
Task 20 – Random Numbers and Monte Carlo Methods
Real-World Case: Estimating the Area of an Irregular Lake (Monte Carlo Integration)
                 and Risk Analysis for a Software Project

PART A – Monte Carlo integration: estimate π and the area of an irregular shape.
PART B – Monte Carlo simulation: software project completion time risk analysis.

Monte Carlo works by:
  1. Defining a domain.
  2. Sampling many random points inside it.
  3. Checking how many satisfy a condition.
  4. Using the ratio to estimate a probability or area.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=123)

# PART A – Estimating π using Monte Carlo
print("=== PART A: Estimating π (Monte Carlo) ===")

def estimate_pi(n, seed=None):
    rng_local = np.random.default_rng(seed)
    x = rng_local.uniform(-1, 1, n)
    y = rng_local.uniform(-1, 1, n)
    inside = (x**2 + y**2) <= 1.0
    return 4 * inside.sum() / n, inside, x, y

sample_sizes = [100, 1000, 10000, 100000, 1000000]
print(f"  {'N':>10}  {'π estimate':>12}  {'Error':>10}")
for n in sample_sizes:
    pi_est, _, _, _ = estimate_pi(n, seed=42)
    print(f"  {n:>10,}  {pi_est:>12.6f}  {abs(pi_est - np.pi):>10.6f}")

# Detailed run for visualisation
n_vis = 2000
pi_est_vis, inside_vis, x_vis, y_vis = estimate_pi(n_vis, seed=42)
print(f"\n  Estimation with n={n_vis}: π ≈ {pi_est_vis:.4f}  (true π = {np.pi:.4f})")


# PART B – Software Project Completion Risk (Monte Carlo Simulation)

print("\n=== PART B: Software Project Risk Analysis ===")

# A software project has 5 tasks. Each task has a minimum and maximum duration.
# We simulate 50 000 possible project outcomes.
N_SIM = 50_000

# Durations in days (min, max) assuming uniform distribution
tasks = {
    "Requirements":  (5,  10),
    "Design":        (8,  15),
    "Development":   (20, 40),
    "Testing":       (5,  12),
    "Deployment":    (2,   5)
}

deadline = 70   # days

total_times = np.zeros(N_SIM)
for task_name, (lo, hi) in tasks.items():
    total_times += rng.uniform(lo, hi, N_SIM)

prob_on_time = (total_times <= deadline).sum() / N_SIM
mean_time    = total_times.mean()
p5           = np.percentile(total_times, 5)
p95          = np.percentile(total_times, 95)

print(f"  Simulations: {N_SIM:,}")
print(f"  Deadline: {deadline} days")
print(f"\n  Mean completion time : {mean_time:.2f} days")
print(f"  5th percentile       : {p5:.2f} days")
print(f"  95th percentile      : {p95:.2f} days")
print(f"  P(finish ≤ {deadline} days) : {prob_on_time:.4f}  ({prob_on_time*100:.1f}%)")

#  Experiment – tighter development phase 
tasks_exp = dict(tasks)
tasks_exp["Development"] = (15, 30)   # development improved
total_exp = np.zeros(N_SIM)
for task_name, (lo, hi) in tasks_exp.items():
    total_exp += rng.uniform(lo, hi, N_SIM)

prob_exp = (total_exp <= deadline).sum() / N_SIM
print(f"\n=== Experiment: Development time reduced to 15–30 days ===")
print(f"  Mean time  : {total_exp.mean():.2f} days  (was {mean_time:.2f})")
print(f"  P(≤ {deadline} days) : {prob_exp:.4f}  ({prob_exp*100:.1f}%)  (was {prob_on_time*100:.1f}%)")
print("  Faster development → significantly higher chance of meeting deadline.")

#  Visualisation 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Monte Carlo π estimation – scatter plot
colors_vis = np.where(inside_vis, "steelblue", "salmon")
axes[0].scatter(x_vis, y_vis, c=colors_vis, s=4, alpha=0.6)
circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1.5)
axes[0].add_patch(circle)
axes[0].set_aspect("equal")
axes[0].set_title(f"Monte Carlo π Estimate ≈ {pi_est_vis:.4f}\n(n={n_vis}, blue=inside circle)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

# π convergence with n
pi_running = []
for n_i in range(100, N_SIM + 1, 100):
    x_r = x_vis[:min(n_i, n_vis)]
    y_r = y_vis[:min(n_i, n_vis)]
    pi_running.append(4 * ((x_r**2 + y_r**2) <= 1).sum() / len(x_r))
axes[1].plot(range(100, len(pi_running)*100 + 1, 100), pi_running, linewidth=0.8, color="purple")
axes[1].axhline(np.pi, color="red", linestyle="--", label=f"True π = {np.pi:.4f}")
axes[1].set_title("π Estimate Convergence")
axes[1].set_xlabel("Number of samples")
axes[1].set_ylabel("π estimate")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

# Project duration histogram
axes[2].hist(total_times, bins=50, color="steelblue",   alpha=0.7, label="Original", density=True)
axes[2].hist(total_exp,   bins=50, color="mediumseagreen", alpha=0.6, label="Experiment", density=True)
axes[2].axvline(deadline, color="red", linestyle="--", linewidth=1.5, label=f"Deadline ({deadline}d)")
axes[2].set_title("Project Completion Time Distribution")
axes[2].set_xlabel("Days")
axes[2].set_ylabel("Density")
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/Task_20_monte_carlo.png", dpi=120)
plt.show()
print("\nPlot saved → results/Task_20_monte_carlo.png")
