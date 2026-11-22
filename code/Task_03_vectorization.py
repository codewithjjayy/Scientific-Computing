"""
BIT 2118: Scientific Computing
Task 03 – Vectorization
Real-World Case: Converting 100 000 Sensor Distance Readings (cm → m)

A warehouse robot logs distances to obstacles using an ultrasonic sensor.
The sensor returns values in centimetres. We must convert all 100 000
readings to metres, clip readings beyond the sensor range (400 cm = 4 m),
and flag readings that are suspiciously close (< 20 cm).

We compare a pure-Python loop with NumPy vectorization to show the
speed difference clearly.

Assumptions:
  - Readings are uniformly distributed between 1 cm and 450 cm.
  - Sensor max reliable range = 400 cm (4.00 m).
  - Minimum safe distance = 20 cm (0.20 m).
"""

import numpy as np
import time
import matplotlib.pyplot as plt

NUM_READINGS = 100_000
rng = np.random.default_rng(seed=7)

# Raw sensor data in centimetres (integer counts)
raw_cm = rng.integers(low=1, high=451, size=NUM_READINGS).astype(float)

#  1. Pure-Python loop (slow) 
t0 = time.perf_counter()
converted_loop = []
for val in raw_cm:
    converted_loop.append(val / 100.0)
t_loop = time.perf_counter() - t0

# 2. NumPy vectorized (fast) 
t0 = time.perf_counter()
converted_np = raw_cm / 100.0          # element-wise division – no loop needed
t_np = time.perf_counter() - t0

speedup = t_loop / t_np
print("=== Vectorization Speed Comparison ===")
print(f"Python loop : {t_loop*1000:.2f} ms")
print(f"NumPy       : {t_np*1000:.4f} ms")
print(f"Speed-up    : {speedup:.0f}×")

#  3. Clip out-of-range readings
MAX_RANGE_M  = 4.00    # metres
MIN_SAFE_M   = 0.20    # metres

clipped_m = np.clip(converted_np, a_min=0.0, a_max=MAX_RANGE_M)

out_of_range = (converted_np > MAX_RANGE_M).sum()
too_close    = (converted_np < MIN_SAFE_M).sum()

print(f"\n=== Sensor Reading Quality ===")
print(f"Total readings     : {NUM_READINGS:,}")
print(f"Out-of-range (>{MAX_RANGE_M} m) : {out_of_range:,}  ({out_of_range/NUM_READINGS*100:.1f}%)")
print(f"Too close    (<{MIN_SAFE_M} m) : {too_close:,}  ({too_close/NUM_READINGS*100:.1f}%)")
print(f"\nSample (first 10 converted, metres):")
print(f"  {converted_np[:10]}")

#  4. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of converted distances
axes[0].hist(clipped_m, bins=50, color="mediumseagreen", edgecolor="white")
axes[0].axvline(MIN_SAFE_M, color="red",    linestyle="--", label=f"Min safe ({MIN_SAFE_M} m)")
axes[0].axvline(MAX_RANGE_M, color="orange", linestyle="--", label=f"Max range ({MAX_RANGE_M} m)")
axes[0].set_title("Distribution of Sensor Readings (metres)")
axes[0].set_xlabel("Distance (m)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Speed comparison bar chart
labels = ["Python loop", "NumPy vectorized"]
times  = [t_loop * 1000, t_np * 1000]
colors = ["salmon", "steelblue"]
axes[1].bar(labels, times, color=colors, edgecolor="white")
axes[1].set_title("Computation Time (ms)")
axes[1].set_ylabel("Time (ms)")
for i, v in enumerate(times):
    axes[1].text(i, v * 1.02, f"{v:.2f} ms", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("results/task_03_vectorization.png", dpi=120)
plt.show()
print("\nPlot saved → task_03_vectorization.png")
