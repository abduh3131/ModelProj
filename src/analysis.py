import numpy as np
import matplotlib.pyplot as plt
import os

from seir_model import get_default_params, GROUP_NAMES, S, E, I, R
from monte_carlo import run_monte_carlo
from interventions import no_intervention, combined_moderate, full_lockdown, DEFAULT_DELAY

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
GROUP_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

# published COVID-19 infection fatality rates by age group
IFR = np.array([0.0001, 0.005, 0.054])  # children 0.01%, adults 0.5%, elderly 5.4%

# published COVID-19 hospitalization rates by age group
HOSP_RATE = np.array([0.01, 0.05, 0.20])  # children 1%, adults 5%, elderly 20%


def compute_severity(results):
    # computes estimated deaths and hospitalizations from infection counts
    num_runs = results["num_runs"]
    num_groups = len(GROUP_NAMES)

    deaths_per_group = np.zeros((num_runs, num_groups))
    hosp_per_group = np.zeros((num_runs, num_groups))

    for i in range(num_runs):
        for g in range(num_groups):
            infected = results["total_infected_per_group"][i, g]
            deaths_per_group[i, g] = infected * IFR[g]
            hosp_per_group[i, g] = infected * HOSP_RATE[g]

    return {
        "deaths_per_group": deaths_per_group,
        "hosp_per_group": hosp_per_group,
        "mean_deaths": np.mean(deaths_per_group, axis=0),
        "mean_hosp": np.mean(hosp_per_group, axis=0),
        "total_mean_deaths": np.mean(deaths_per_group.sum(axis=1)),
        "total_mean_hosp": np.mean(hosp_per_group.sum(axis=1)),
    }


def plot_severity_comparison(scenario_results_list):
    # bar chart showing estimated deaths and hospitalizations per scenario
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scenario_names = [name for name, _ in scenario_results_list]
    x = np.arange(len(scenario_names))
    bar_width = 0.25

    for g in range(len(GROUP_NAMES)):
        deaths = []
        hosps = []
        for name, results in scenario_results_list:
            sev = compute_severity(results)
            deaths.append(sev["mean_deaths"][g])
            hosps.append(sev["mean_hosp"][g])

        offset = (g - 1) * bar_width
        ax1.bar(x + offset, deaths, bar_width, label=GROUP_NAMES[g],
                color=GROUP_COLORS[g], edgecolor="black", linewidth=0.4)
        ax2.bar(x + offset, hosps, bar_width, label=GROUP_NAMES[g],
                color=GROUP_COLORS[g], edgecolor="black", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Estimated Deaths")
    ax1.set_title("Estimated Deaths by Group and Scenario", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Estimated Hospitalizations")
    ax2.set_title("Estimated Hospitalizations by Group and Scenario", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "severity_comparison.png"), dpi=150)
    plt.close()


def plot_detection_delay_comparison():
    # compares total infections for different detection delays
    delays = [0, 7, 14, 21, 28]
    scenarios_to_test = [
        ("Combined Moderate", combined_moderate),
        ("Full Lockdown", full_lockdown),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#f39c12", "#9b59b6"]

    for idx, (label, scenario_fn) in enumerate(scenarios_to_test):
        means = []
        stds = []
        for d in delays:
            params, _ = scenario_fn(start_day=d)
            results = run_monte_carlo(params, num_runs=100, base_seed=42)
            means.append(np.mean(results["total_infected_per_run"]))
            stds.append(np.std(results["total_infected_per_run"]))

        ax.errorbar(delays, means, yerr=stds, marker="o", linewidth=2,
                    capsize=5, label=label, color=colors[idx])

    # add no intervention baseline as horizontal line
    params_base, _ = no_intervention()
    results_base = run_monte_carlo(params_base, num_runs=100, base_seed=42)
    baseline_mean = np.mean(results_base["total_infected_per_run"])
    ax.axhline(y=baseline_mean, color="#95a5a6", linestyle="--", linewidth=1.5, label="No Intervention")

    ax.set_xlabel("Detection Delay (days)")
    ax.set_ylabel("Total Infections")
    ax.set_title("Impact of Detection Delay on Intervention Effectiveness", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(delays)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "detection_delay_comparison.png"), dpi=150)
    plt.close()

    return delays


def plot_r0_sensitivity():
    # tests how different R0 values affect the outbreak
    # R0 = beta * contacts * infectious_period, we vary beta to change R0
    beta_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]
    r0_labels = []

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#95a5a6", "#e74c3c", "#8e44ad"]

    for idx, mult in enumerate(beta_multipliers):
        params = get_default_params()
        params["beta"] = params["beta"] * mult

        # estimate effective R0 for the label
        avg_beta = np.mean(params["beta"])
        avg_contacts = np.mean(params["contact_matrix"])
        avg_gamma = np.mean(params["gamma"])
        approx_r0 = avg_beta * avg_contacts * (1.0 / avg_gamma)
        r0_labels.append(f"R0 ~ {approx_r0:.1f}")

        results = run_monte_carlo(params, num_runs=100, base_seed=42)
        mean_I = results["mean"][:, :, I].sum(axis=1)
        days = np.arange(len(mean_I))

        ax.plot(days, mean_I, label=f"R0 ~ {approx_r0:.1f} (beta x{mult})",
                color=colors[idx], linewidth=2)

    ax.set_xlabel("Days")
    ax.set_ylabel("Total Active Infections")
    ax.set_title("R0 Sensitivity Analysis: Effect on Infection Curve", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "r0_sensitivity.png"), dpi=150)
    plt.close()


def plot_monte_carlo_distribution(results, scenario_name):
    # histogram of total infections across all Monte Carlo runs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results["total_infected_per_run"], bins=30, color="#3498db",
            edgecolor="black", alpha=0.7)

    mean_val = np.mean(results["total_infected_per_run"])
    ax.axvline(x=mean_val, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:,.0f}")

    ci_low = np.percentile(results["total_infected_per_run"], 5)
    ci_high = np.percentile(results["total_infected_per_run"], 95)
    ax.axvline(x=ci_low, color="#f39c12", linestyle=":", linewidth=1.5,
               label=f"5th percentile: {ci_low:,.0f}")
    ax.axvline(x=ci_high, color="#f39c12", linestyle=":", linewidth=1.5,
               label=f"95th percentile: {ci_high:,.0f}")

    ax.set_xlabel("Total Infections")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo Distribution - {scenario_name} ({results['num_runs']} runs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(OUTPUT_DIR, f"mc_distribution_{safe_name}.png"), dpi=150)
    plt.close()


def print_severity_table(scenario_results_list):
    # prints mortality and hospitalization estimates
    print("\n" + "=" * 95)
    print(f"{'Scenario':<35} {'Deaths':>10} {'Hospitalizations':>18} {'Death Rate':>12}")
    print("=" * 95)
    for name, results in scenario_results_list:
        sev = compute_severity(results)
        total_pop = results["params"]["population"].sum()
        print(f"{name:<35} {sev['total_mean_deaths']:>10,.0f} "
              f"{sev['total_mean_hosp']:>18,.0f} "
              f"{sev['total_mean_deaths']/total_pop*100:>11.3f}%")

    print("=" * 95)

    # per-group breakdown for baseline
    print("\n  Per-group severity breakdown (No Intervention baseline):")
    print(f"  {'Group':<25} {'Deaths':>10} {'Hospitalizations':>18} {'IFR':>8} {'Hosp Rate':>10}")
    print(f"  {'-'*75}")
    _, baseline_results = scenario_results_list[0]
    sev = compute_severity(baseline_results)
    for g in range(len(GROUP_NAMES)):
        print(f"  {GROUP_NAMES[g]:<25} {sev['mean_deaths'][g]:>10,.0f} "
              f"{sev['mean_hosp'][g]:>18,.0f} "
              f"{IFR[g]*100:>7.2f}% "
              f"{HOSP_RATE[g]*100:>9.0f}%")
    print(f"  {'TOTAL':<25} {sev['total_mean_deaths']:>10,.0f} "
          f"{sev['total_mean_hosp']:>18,.0f}")


def run_all_analysis(scenario_results_list):
    # runs all additional analysis and generates plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Additional Analysis ---")

    print("  Generating severity comparison...")
    plot_severity_comparison(scenario_results_list)
    print("  - Severity comparison saved")

    print("  Generating Monte Carlo distribution (baseline)...")
    plot_monte_carlo_distribution(scenario_results_list[0][1], scenario_results_list[0][0])
    print("  - MC distribution saved")

    print("  Running detection delay comparison (this takes a minute)...")
    plot_detection_delay_comparison()
    print("  - Detection delay comparison saved")

    print("  Running R0 sensitivity analysis...")
    plot_r0_sensitivity()
    print("  - R0 sensitivity saved")

    print_severity_table(scenario_results_list)
