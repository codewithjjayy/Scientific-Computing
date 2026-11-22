peak_infected = sol_sir.y[1].max()
t_peak = sol_sir.t[np.argmax(sol_sir.y[1])]
print(f"Peak infections: {peak_infected:.0f} people at t = {t_peak:.1f} days")
print(f"Total recovered: {sol_sir.y[2][-1]:.0f} people")

# Extension 19: Minimum-Cost Box Design
print("\nExtension 19: Optimal Box Dimensions (Min Cost)")

# Open-top box with square base (volume = 500 cm³)
# Base cost = $2/cm², side cost = $1/cm²
def box_cost(l):
    h = 500 / l**2
    base_area = l**2
    side_area = 4 * l * h
    return 2 * base_area + side_area

res_box = minimize_scalar(box_cost, bounds=(1, 50), method="bounded")
l_opt = res_box.x
h_opt = 500 / l_opt**2

print(f"Optimal base length: {l_opt:.4f} cm")
print(f"Optimal height: {h_opt:.4f} cm")
print(f"Minimum cost: ${res_box.fun:.4f}")
print(f"Volume check: {l_opt**2 * h_opt:.2f} cm³ (target: 500)")

# Extension 20: Monte Carlo Network Packet Loss Simulation
print("\nExtension 20: Monte Carlo - Network Packet Loss Simulation")

# Simulate 3 network links in series (each fails independently with p = 0.05)
p_fail = 0.05
n_trials = 100_000
rng = np.random.default_rng(seed=200)

link1 = rng.uniform(0, 1, n_trials) < p_fail
link2 = rng.uniform(0, 1, n_trials) < p_fail
link3 = rng.uniform(0, 1, n_trials) < p_fail

connection_failed = link1 | link2 | link3
p_connection_fail = connection_failed.mean()
p_exact = 1 - (1 - p_fail)**3

print(f"Trials: {n_trials:,}")
print(f"P(connection fails) - Monte Carlo: {p_connection_fail:.4f}")
print(f"P(connection fails) - Exact: {p_exact:.4f}")
print(f"Difference: {abs(p_connection_fail - p_exact):.4f}")

print("\nAll individual extensions completed.")
