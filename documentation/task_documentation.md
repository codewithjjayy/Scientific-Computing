# BIT 2118 Scientific Computing – Task Documentation

This document explains each task: the concept, the formula, the code, and what the output means.

---

## Task 02 – NumPy Arrays

**Concept:**  
An array stores many numbers in a grid so you can do calculations on all of them at once instead of writing loops.

**Formula:**  
`mean = sum(values) / count`  — NumPy does this across rows or columns automatically.

**Real-world application:**  
Storing sensor readings, stock prices, image pixels.

**Code file:** `code/Task_02_numpy_arrays.py`  
**What it does:** Creates a 10×7 matrix of solar panel energy readings. Finds daily totals, panel averages, and flags low-output readings. Plots a heatmap and bar chart.

**Key lines:**
- `energy.sum(axis=0)` — sums across rows (all panels) for each day
- `energy < threshold` — Boolean mask: True where value is below threshold
- `imshow(energy)` — displays the matrix as a colour heatmap

**Experiment:** Changing the threshold from 4.5 to 6.0 kWh increases the number of "low output" readings because more cells fall below the new limit.

---

## Task 03 – Vectorization

**Concept:**  
Vectorization means applying an operation to an entire array at once (no Python loop). NumPy uses optimised C code internally, making it much faster.

**Formula:**  
`v_metres = v_centimetres / 100`  — applied to 100 000 values in one line.

**Code file:** `code/Task_03_vectorization.py`  
**What it does:** Converts 100 000 robot sensor distances from cm to m. Times the loop vs NumPy version. Clips out-of-range readings.

**Key lines:**
- `raw_cm / 100.0` — vectorized division (no loop)
- `np.clip(arr, a_min, a_max)` — limits values to a range
- `time.perf_counter()` — high-resolution timer

**Experiment:** With 1 000 000 readings the speedup is even larger (NumPy is ~50–100× faster than a Python loop).

---

## Task 04 – Floating-Point Numerical Errors

**Concept:**  
Computers store numbers in binary. Some decimal fractions (like 0.1) cannot be represented exactly, so small errors accumulate over many calculations.

**Formula:**  
`0.1 + 0.2 ≠ 0.3` in standard float arithmetic.

**Code file:** `code/Task_04_floating_point_errors.py`  
**What it does:** Demonstrates 0.1+0.2 ≠ 0.3, accumulation over 10 000 additions, catastrophic cancellation, and shows `math.fsum` and `Decimal` as fixes.

**Key lines:**
- `math.fsum([0.10]*10000)` — compensated summation, much more accurate
- `Decimal("0.10")` — exact decimal arithmetic (no binary representation error)

---

## Task 05 – Matrix Algebra

**Concept:**  
A matrix is a 2D grid of numbers. Matrix operations include addition (element by element), scalar multiplication, transpose (flip rows/columns), and matrix multiplication.

**Formula:**  
`C = A @ B` — matrix multiplication (dot product of rows and columns)  
`det(A)` — determinant; if zero, the matrix has no inverse

**Code file:** `code/Task_05_matrix_algebra.py`  
**What it does:** Models router link costs as 4×4 matrices. Demonstrates addition, scalar multiplication, transpose, matrix multiplication, determinant, and inverse. Plots heatmaps.

**Key lines:**
- `A @ A` — matrix multiplication (two-hop reachability)
- `np.linalg.det(M)` — determinant
- `np.linalg.inv(M)` — matrix inverse
- `np.allclose(A, A.T)` — checks if A is symmetric (undirected graph)

---

## Task 06 – Solution of Systems of Linear Equations

**Concept:**  
When you have N unknowns and N equations, you can write them as `A·x = b` and solve for x. This appears in circuits, structures, and economic models.

**Formula:**  
`x = A⁻¹ · b`  (inverse method)  
Cramer's Rule: `xᵢ = det(Aᵢ) / det(A)`

**Code file:** `code/Task_06_linear_equations.py`  
**What it does:** Finds three branch currents in an electrical circuit using Kirchhoff's Voltage Law. Solves with np.linalg.solve, Cramer's Rule, and the matrix inverse.

**Possible lecturer question:** *Why use `np.linalg.solve` instead of the inverse?*  
Answer: `solve` is more numerically stable and faster than explicitly computing the inverse.

---

## Task 07 – Gaussian Elimination and LU Decomposition

**Concept:**  
Gaussian Elimination transforms the system into an upper triangular form, then uses back-substitution. LU Decomposition factors A = P·L·U, which is especially efficient when solving many systems with the same A.

**Code file:** `code/Task_07_gaussian_lu.py`  
**What it does:** Finds reaction forces in a loaded beam. Implements Gaussian elimination manually (with partial pivoting), then repeats using `scipy.linalg.lu`.

**Key lines:**
- Partial pivoting: `M[[col, max_row]] = M[[max_row, col]]` — swaps rows so the largest element is on the diagonal (improves stability)
- `lu_factor(A)` + `lu_solve(lu_fac, b)` — SciPy's efficient LU solver

---

## Task 08 – Iterative Methods (Jacobi and Gauss-Seidel)

**Concept:**  
Instead of solving `A·x = b` directly, start with a guess x=0 and repeatedly improve it. Jacobi uses the old values for all updates; Gauss-Seidel uses new values as soon as they are computed (converges faster).

**Formula:**  
Jacobi: `xᵢ(k+1) = (bᵢ - Σⱼ≠ᵢ aᵢⱼ xⱼ(k)) / aᵢᵢ`

**Code file:** `code/Task_08_iterative_methods.py`  
**What it does:** Finds steady-state temperatures at 4 interior points of a metal rod. Compares Jacobi and Gauss-Seidel convergence.

**Experiment:** Gauss-Seidel always converges in fewer iterations than Jacobi for this diagonally dominant system.

---

## Task 09 – Eigenvalues and Eigenvectors

**Concept:**  
An eigenvector of matrix A is a special vector that only gets scaled (not rotated) when multiplied by A. The scale factor is the eigenvalue λ. `A·v = λ·v`.

**Real-world use:** Natural frequencies of structures, principal components in data, Google's PageRank algorithm.

**Code file:** `code/Task_09_eigenvalues.py`  
**What it does:** Computes vibration modes of a 2-storey building. Each eigenvalue → natural frequency. Each eigenvector → shape of that vibration mode.

---

## Task 10 – Bisection and Fixed-Point Methods

**Concept:**  
Root finding means finding x where f(x) = 0.  
Bisection: repeatedly halve an interval [a,b] where f changes sign.  
Fixed-Point: rewrite f(x)=0 as x=g(x) and iterate.

**Code file:** `code/Task_10_bisection_fixedpoint.py`  
**What it does:** Finds the number of units where a business breaks even (profit = 0).

**Key point:** Bisection always converges if f(a) and f(b) have opposite signs. Fixed-point only converges if |g'(x)| < 1 near the root.

---

## Task 11 – Newton-Raphson and Secant Methods

**Concept:**  
Newton-Raphson uses the function value AND its derivative to jump to the root quickly (quadratic convergence). Secant approximates the derivative using two previous points (useful when derivative is hard to compute).

**Formula:**  
Newton-Raphson: `xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)`  
Secant: `xₙ₊₁ = xₙ - f(xₙ)·(xₙ-xₙ₋₁) / (f(xₙ)-f(xₙ₋₁))`

**Code file:** `code/Task_11_newton_secant.py`  
**What it does:** Finds the pipe radius that delivers a target flow rate.

---

## Task 12 – Polynomial and Nonlinear Equations

**Concept:**  
`numpy.roots` finds all roots of a polynomial. `scipy.optimize.brentq` finds a root of any continuous function in an interval. `fsolve` solves systems of nonlinear equations.

**Code file:** `code/Task_12_polynomial_nonlinear.py`  
**What it does:** Finds launch angles for a target projectile range. Also solves for circle-sine intersection using `fsolve`.

---

## Task 13 – Interpolation

**Concept:**  
Interpolation estimates values between known data points. Linear: straight lines between points. Polynomial: one curve through all points. Cubic Spline: smooth piecewise-cubic curves (best for data with smooth trends).

**Code file:** `code/Task_13_interpolation.py`  
**What it does:** Estimates missing hourly temperatures. Compares linear, polynomial, and cubic spline methods.

**Possible lecturer question:** *Why is a high-degree polynomial sometimes bad?*  
Answer: Runge's phenomenon — high-degree polynomials can oscillate wildly between data points.

---

## Task 14 – Curve Fitting

**Concept:**  
Unlike interpolation, curve fitting does NOT require passing through every point. It finds the best-fitting function (minimising squared errors). R² measures how well the curve fits (1.0 = perfect).

**Code file:** `code/Task_14_curve_fitting.py`  
**What it does:** Fits linear, quadratic, and exponential models to a battery discharge curve. Compares R² values.

---

## Task 15 – Numerical Differentiation

**Concept:**  
When you only have data (no formula), estimate the derivative using finite differences.  
Central difference is most accurate because the O(h²) error is much smaller than the O(h) error of forward/backward difference.

**Code file:** `code/Task_15_differentiation.py`  
**What it does:** Estimates vehicle velocity and acceleration from GPS position data. Compares three finite-difference formulas against the exact derivative.

---

## Task 16 – Numerical Integration

**Concept:**  
When you cannot integrate a function analytically, approximate the area under the curve using the Trapezoidal Rule, Simpson's Rule, or Gaussian Quadrature.

**Code file:** `code/Task_16_integration.py`  
**What it does:** Calculates total distance travelled by a train from its speed-time curve.

**Key point:** Simpson's Rule is more accurate than the Trapezoidal Rule for the same number of points because it fits parabolas instead of straight lines.

---

## Task 17 – Euler's Method (ODE)

**Concept:**  
An ODE gives a rule for how something changes: dy/dt = f(t,y). Euler's method steps forward in small increments h, using the current slope to estimate the next value.

**Formula:**  
`y_{n+1} = y_n + h · f(t_n, y_n)`

**Code file:** `code/Task_17_euler_ode.py`  
**What it does:** Models a cooling cup of coffee. Compares step sizes h=5, h=1, h=0.5 against the exact analytic solution.

---

## Task 18 – SciPy ODE Solver

**Concept:**  
`scipy.integrate.solve_ivp` with method `RK45` (Runge-Kutta 4/5) is much more accurate than Euler's method. It automatically controls step size to meet the requested tolerance.

**Code file:** `code/Task_18_scipy_ode.py`  
**What it does:** Solves the Lotka-Volterra predator-prey equations. Plots population over time and a phase portrait showing the cyclic relationship.

---

## Task 19 – Numerical Optimisation

**Concept:**  
Finding the input value that minimises (or maximises) a function. `minimize_scalar` for one variable, `minimize` for multiple variables, with or without constraints.

**Code file:** `code/Task_19_optimization.py`  
**What it does:** Finds the production rate that minimises daily cost. Also demonstrates 2D optimisation and constrained profit maximisation.

---

## Task 20 – Monte Carlo Methods

**Concept:**  
Monte Carlo uses many random samples to estimate probabilities or integrals. The estimate gets better as N increases (error ∝ 1/√N).

**Code file:** `code/Task_20_monte_carlo.py`  
**What it does:**  
- Part A: Estimates π by sampling random points and checking if they fall inside a unit circle.  
- Part B: Simulates 50 000 possible project durations to estimate the probability of meeting a deadline.

---

## AI Assistance Disclosure

This assignment was developed with assistance from an AI programming tool (IBM Bob).  
The mathematical concepts, formulas, code structure, and explanations were reviewed and are understood by the student.  
All programs produce correct outputs based on the stated mathematical methods.
