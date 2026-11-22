# BIT 2118 – Scientific Computing

**Course:** BIT 2118 Scientific Computing  
**Repository owner:** [codewithjjayy](https://github.com/codewithjjayy)

---

## Project Description

This repository contains my individual solutions for the BIT 2118 Scientific Computing assignment.  
Each task covers a core numerical computing concept, implemented in Python using NumPy, SciPy and Matplotlib.

The work follows the official assignment structure: a mathematical explanation, a real-world problem, working Python code, results, an experiment with changed inputs, and an individual real-world extension.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Arrays, linear algebra, numerical operations |
| `scipy` | LU decomposition, ODE solver, optimisation, integration, interpolation |
| `matplotlib` | Graphs and visualisations |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
BIT2118-Scientific-Computing/
├── code/
│   ├── Task_02_numpy_arrays.py
│   ├── Task_03_vectorization.py
│   ├── Task_04_floating_point_errors.py
│   ├── Task_05_matrix_algebra.py
│   ├── Task_06_linear_equations.py
│   ├── Task_07_gaussian_lu.py
│   ├── Task_08_iterative_methods.py
│   ├── Task_09_eigenvalues.py
│   ├── Task_10_bisection_fixedpoint.py
│   ├── Task_11_newton_secant.py
│   ├── Task_12_polynomial_nonlinear.py
│   ├── Task_13_interpolation.py
│   ├── Task_14_curve_fitting.py
│   ├── Task_15_differentiation.py
│   ├── Task_16_integration.py
│   ├── Task_17_euler_ode.py
│   ├── Task_18_scipy_ode.py
│   ├── Task_19_optimization.py
│   └── Task_20_monte_carlo.py
├── individual_extensions/
│   └── all_extensions.py
├── documentation/
│   └── task_documentation.md
├── results/          ← graphs saved here when scripts run
├── README.md
└── requirements.txt
```

---

## Running the Scripts

Run any task from the project root:

```bash
cd BIT2118-Scientific-Computing
python code/Task_05_matrix_algebra.py
```

Graphs are saved automatically to the `results/` folder.  
Make sure the `results/` folder exists before running (it is included in this repo).

---

## Task Summary

| Task | Topic | Real-World Case |
|------|-------|----------------|
| 02 | NumPy Arrays | Solar panel energy readings |
| 03 | Vectorization | Warehouse robot sensor data |
| 04 | Floating-Point Errors | Financial transaction rounding |
| 05 | Matrix Algebra | Network router link costs |
| 06 | Linear Equations | Circuit branch currents (KVL) |
| 07 | Gaussian Elimination & LU | Beam support reaction forces |
| 08 | Iterative Methods (Jacobi, Gauss-Seidel) | Temperature in a metal rod |
| 09 | Eigenvalues & Eigenvectors | Building vibration mode shapes |
| 10 | Bisection & Fixed-Point | Business break-even point |
| 11 | Newton-Raphson & Secant | Optimal pipe radius |
| 12 | Polynomial & Nonlinear Equations | Projectile launch angles |
| 13 | Interpolation | Missing temperature readings |
| 14 | Curve Fitting | Smartphone battery discharge |
| 15 | Numerical Differentiation | Vehicle instantaneous speed |
| 16 | Numerical Integration | Train total distance travelled |
| 17 | Euler ODE Method | Coffee cooling (Newton's Law) |
| 18 | SciPy ODE Solver | Predator-prey (Lotka-Volterra) |
| 19 | Numerical Optimisation | Minimum production cost |
| 20 | Monte Carlo Methods | Estimating π and project risk |

---

## Individual Extensions

All individual extensions are in `individual_extensions/all_extensions.py`.  
Each extension applies the same mathematical concept from the original task to a different real-world scenario:

| Task | Extension Scenario |
|------|--------------------|
| 02 | Retail store weekly sales analysis |
| 03 | Batch student grade conversion |
| 04 | Bank compound interest accumulation error |
| 05 | Student final mark computation (weighted) |
| 06 | Nutritional meal planning |
| 07 | Traffic signal timing equations |
| 08 | Pressure in a pipeline network |
| 09 | Sensor covariance dominant directions |
| 10 | Bond yield calculation |
| 11 | Compound interest rate solver |
| 12 | Fuel consumption vs driving speed |
| 13 | Stock price interpolation |
| 14 | Internet usage growth (linear fit) |
| 15 | Marginal cost in economics |
| 16 | Work done by a variable force |
| 17 | Drug concentration in bloodstream |
| 18 | SIR epidemic model |
| 19 | Optimal box dimensions (min cost) |
| 20 | Network packet loss probability |

---

## Notes

- All outputs shown in the documentation were produced by actually running the scripts.
- AI assistance was used in developing this work. This is documented transparently in accordance with institutional requirements.
