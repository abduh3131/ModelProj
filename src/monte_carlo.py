import numpy as np
from seir_model import run_simulation


def run_monte_carlo(params, num_runs=300, base_seed=42):
    # runs multiple SEIR simulations and aggregates results
    first_run = run_simulation(params, seed=base_seed)
    num_steps_plus_one = first_run.shape[0]
    num_groups = first_run.shape[1]

    all_runs = np.zeros((num_runs, num_steps_plus_one, num_groups, 4))
    all_runs[0] = first_run

    for i in range(1, num_runs):
        all_runs[i] = run_simulation(params, seed=base_seed + i)

    # compute statistics across runs
    mean = np.mean(all_runs, axis=0)
    median = np.median(all_runs, axis=0)
    ci_lower = np.percentile(all_runs, 5, axis=0)
    ci_upper = np.percentile(all_runs, 95, axis=0)

    # compute summary metrics per run
    total_infected_per_run = np.zeros(num_runs)
    peak_infected_per_run = np.zeros(num_runs)
    peak_day_per_run = np.zeros(num_runs, dtype=int)

    # per-group metrics: (num_runs, num_groups)
    total_infected_per_group = np.zeros((num_runs, num_groups))
    peak_infected_per_group = np.zeros((num_runs, num_groups))
    peak_day_per_group = np.zeros((num_runs, num_groups), dtype=int)
    infection_start_day_per_group = np.zeros((num_runs, num_groups), dtype=int)
    infection_end_day_per_group = np.zeros((num_runs, num_groups), dtype=int)

    for i in range(num_runs):
        # total infected = initial S - final S
        initial_s = all_runs[i, 0, :, 0].sum()
        final_s = all_runs[i, -1, :, 0].sum()
        total_infected_per_run[i] = initial_s - final_s

        # peak active infections across all groups
        total_I_over_time = all_runs[i, :, :, 2].sum(axis=1)
        peak_infected_per_run[i] = total_I_over_time.max()
        peak_day_per_run[i] = total_I_over_time.argmax()

        # per-group breakdown
        for g in range(num_groups):
            initial_s_g = all_runs[i, 0, g, 0]
            final_s_g = all_runs[i, -1, g, 0]
            total_infected_per_group[i, g] = initial_s_g - final_s_g

            I_over_time_g = all_runs[i, :, g, 2]
            peak_infected_per_group[i, g] = I_over_time_g.max()
            peak_day_per_group[i, g] = I_over_time_g.argmax()

            # infection start: first day where I > 1% of group peak
            threshold = I_over_time_g.max() * 0.01
            above = np.where(I_over_time_g > threshold)[0]
            infection_start_day_per_group[i, g] = above[0] if len(above) > 0 else 0

            # infection end: last day where I > 1% of group peak
            infection_end_day_per_group[i, g] = above[-1] if len(above) > 0 else num_steps_plus_one - 1

    results = {
        "all_runs": all_runs,
        "mean": mean,
        "median": median,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "total_infected_per_run": total_infected_per_run,
        "peak_infected_per_run": peak_infected_per_run,
        "peak_day_per_run": peak_day_per_run,
        "total_infected_per_group": total_infected_per_group,
        "peak_infected_per_group": peak_infected_per_group,
        "peak_day_per_group": peak_day_per_group,
        "infection_start_day_per_group": infection_start_day_per_group,
        "infection_end_day_per_group": infection_end_day_per_group,
        "num_runs": num_runs,
        "params": params,
    }

    return results


def compute_scenario_comparison(scenario_results_list):
    # compares total infections across intervention scenarios
    comparison = {}

    for name, results in scenario_results_list:
        total_pop = results["params"]["population"].sum()
        comparison[name] = {
            "mean_total_infected": np.mean(results["total_infected_per_run"]),
            "std_total_infected": np.std(results["total_infected_per_run"]),
            "mean_attack_rate": np.mean(results["total_infected_per_run"]) / total_pop,
            "mean_peak_infected": np.mean(results["peak_infected_per_run"]),
            "mean_peak_day": np.mean(results["peak_day_per_run"]),
            "ci90_total_infected": (
                np.percentile(results["total_infected_per_run"], 5),
                np.percentile(results["total_infected_per_run"], 95),
            ),
        }

    return comparison
