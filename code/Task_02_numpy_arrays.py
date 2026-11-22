"""
BIT 2118: Scientific Computing
Task 02 – NumPy Arrays
Real-World Case: Daily Solar Panel Energy Readings

A solar farm records energy output (in kWh) for each of its 10 panels
over a 7-day week. We use NumPy arrays to store, inspect, and summarise
the data without writing slow Python loops.

Assumptions:
  - Each panel produces between 3 kWh and 9 kWh per day (randomly seeded
    for reproducibility).
  - Rows = panels (10), Columns = days (7).
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Create the data ────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)          # fixed seed → same numbers every run
energy = rng.uniform(low=3.0, high=9.0, size=(10, 7))   # shape: (panels, days)

days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
panels = [f"Panel-{i+1}" for i in range(10)]

# ── 2. Basic array inspection ─────────────────────────────────────────────────
print("=== Solar Panel Energy Readings (kWh) ===")
print(f"Array shape : {energy.shape}   (panels × days)")
print(f"Data type   : {energy.dtype}")
print(f"\nFirst row (Panel-1 readings):\n  {energy[0]}")

# ── 3. Summary statistics ─────────────────────────────────────────────────────
daily_total  = energy.sum(axis=0)      # total energy per day  (shape: 7)
panel_avg    = energy.mean(axis=1)     # average daily output per panel (shape: 10)
overall_avg  = energy.mean()
best_day_idx = daily_total.argmax()

print("\n=== Summary Statistics ===")
print(f"Overall average output : {overall_avg:.2f} kWh")
print(f"Best production day    : {days[best_day_idx]}  ({daily_total[best_day_idx]:.2f} kWh total)")
print(f"\nPer-panel average (kWh/day):")
for name, avg in zip(panels, panel_avg):
    print(f"  {name}: {avg:.2f}")

# ── 4. Boolean masking – find under-performing readings ───────────────────────
threshold  = 4.5                        # kWh; below this = poor output
low_output = energy < threshold
print(f"\nReadings below {threshold} kWh: {low_output.sum()} out of {energy.size}")

# ── 5. Visualisation ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Heat-map of all readings
im = axes[0].imshow(energy, aspect="auto", cmap="YlOrRd")
axes[0].set_title("Energy Output Heat-Map (kWh)")
axes[0].set_xticks(range(7));  axes[0].set_xticklabels(days)
axes[0].set_yticks(range(10)); axes[0].set_yticklabels(panels)
plt.colorbar(im, ax=axes[0], label="kWh")

# Bar chart of daily totals
axes[1].bar(days, daily_total, color="steelblue", edgecolor="white")
axes[1].set_title("Total Daily Energy Output (All Panels)")
axes[1].set_ylabel("kWh")
axes[1].set_xlabel("Day of Week")
for i, v in enumerate(daily_total):
    axes[1].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("results/task_02_solar_energy.png", dpi=120)
plt.show()
print("\nPlot saved → task_02_solar_energy.png")
