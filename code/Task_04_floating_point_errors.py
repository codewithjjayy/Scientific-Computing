"""
BIT 2118: Scientific Computing
Task 04 – Floating-Point Numerical Errors
Real-World Case: Financial / Currency Rounding Error Check

A cashier system processes 10 000 small transactions. Each transaction
amount is stored in floating-point format. After summing all amounts,
the system compares the computed total against an exact integer total
to detect rounding drift – a known problem in financial software.

We also demonstrate catastrophic cancellation (subtracting two nearly
equal numbers) and show how to mitigate it with math.fsum() or the
Decimal module.

Assumptions:
  - Each transaction is between $0.01 and $9.99 (2 decimal places ideally).
  - We use $0.10 repeated additions to trigger the classic 0.1 + 0.2 ≠ 0.3 error.
"""

import numpy as np
import math
from decimal import Decimal, getcontext

#  1. Classic floating-point demo: 0.1 + 0.2 ≠ 0.3 
a = 0.1
b = 0.2
c = a + b
print("=== Classic Floating-Point Demo ===")
print(f"  0.1 + 0.2        = {c}")          # prints 0.30000000000000004
print(f"  0.1 + 0.2 == 0.3 : {c == 0.3}")  # False!
print(f"  Difference       : {abs(c - 0.3):.2e}")

#  2. Accumulation error over 10 000 transactions 
NUM_TX   = 10_000
amount   = 0.10                        # each transaction = $0.10 exactly (in decimal)
exact_total = NUM_TX * Decimal("0.10") # exact answer using Decimal arithmetic

# Float loop sum
float_sum = 0.0
for _ in range(NUM_TX):
    float_sum += amount

# numpy sum
np_sum = np.sum(np.full(NUM_TX, 0.10))

# math.fsum (compensated summation – much more accurate)
fsum = math.fsum([0.10] * NUM_TX)

# Decimal sum (exact)
dec_sum = sum(Decimal("0.10") for _ in range(NUM_TX))

print("\n=== Summing $0.10 × 10 000 Transactions ===")
print(f"  Exact answer (Decimal) : ${exact_total}")
print(f"  Python float loop      : ${float_sum:.6f}  error = {abs(float_sum - float(exact_total)):.2e}")
print(f"  NumPy np.sum           : ${np_sum:.6f}  error = {abs(np_sum   - float(exact_total)):.2e}")
print(f"  math.fsum              : ${fsum:.6f}  error = {abs(fsum     - float(exact_total)):.2e}")
print(f"  Decimal (exact)        : ${dec_sum}")

# 3. Catastrophic cancellation demo 
# Imagine measuring voltage: V1 = 1.0000001 V, V2 = 1.0000000 V
# Difference should be 0.0000001 but floating point loses precision
V1 = 1.0000001
V2 = 1.0000000
true_diff = Decimal("1.0000001") - Decimal("1.0000000")  # = 0.0000001

fp_diff   = V1 - V2
rel_error = abs(float(fp_diff) - float(true_diff)) / float(true_diff) * 100

print("\n=== Catastrophic Cancellation (Voltage Sensor) ===")
print(f"  V1 - V2 (float)   : {fp_diff:.10f}")
print(f"  True difference   : {true_diff}")
print(f"  Relative error    : {rel_error:.4f}%")

#  4. Visualisation 
import matplotlib.pyplot as plt

n_steps = np.arange(1, NUM_TX + 1, step=100)
running_errors = []
running = 0.0
for i in range(NUM_TX):
    running += 0.10
    if (i + 1) % 100 == 0:
        running_errors.append(abs(running - (i + 1) * 0.10))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(n_steps, running_errors, color="crimson", linewidth=1.2)
ax.set_title("Accumulated Floating-Point Error vs. Number of $0.10 Additions")
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Absolute Error ($)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.savefig("results/task_04_floating_point.png", dpi=120)
plt.show()
print("\nPlot saved → task_04_floating_point.png")
