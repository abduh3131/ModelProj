import os
import numpy as np
import matplotlib.pyplot as plt

from seir_model import run_simulation, get_default_params, GROUP_NAMES, S, E, I, R
from monte_carlo import run_monte_carlo, compute_scenario_comparison
from interventions import get_all_scenarios, DEFAULT_DELAY

GROUP_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def plot_single_scenario(results, scenario_name):
    # SEIR curves w 90% CI for one scenario
    days = np.arange(results["mean"].shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"SEIR Dynamics - {scenario_name} ({results['num_runs']} Monte Carlo runs)", fontsize=13)

    colors = {"S": "#3498db", "E": "#f39c12", "I": "#e74c3c", "R": "#2ecc71"}
    start_day = results["params"].get("intervention_start_day", 0)

    for g, ax in enumerate(axes):
        for comp, label, color in [(S, "S", colors["S"]), (E, "E", colors["E"]),
                                    (I, "I", colors["I"]), (R, "R", colors["R"])]:
            ax.plot(days, results["mean"][:, g, comp], label=label, color=color, linewidth=1.5)
            ax.fill_between(days, results["ci_lower"][:, g, comp],
                           results["ci_upper"][:, g, comp], alpha=0.15, color=color)

        if start_day > 0:
            ax.axvline(x=start_day, color="black", linestyle="--", alpha=0.5,
                      label=f"Intervention (day {start_day})")

        ax.set_title(GROUP_NAMES[g], fontsize=11)
        ax.set_xlabel("Days")
        if g == 0:
            ax.set_ylabel("Population")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(OUTPUT_DIR, f"seir_{safe_name}.png"), dpi=150)
    plt.close()


def plot_scenario_comparison(scenario_results_list):
    # bar chart total infections all scenarios
    comparison = compute_scenario_comparison(scenario_results_list)
    names = list(comparison.keys())
    means = [comparison[n]["mean_total_infected"] for n in names]
    stds = [comparison[n]["std_total_infected"] for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total Infections (mean +/- std)")
    ax.set_title("Intervention Scenario Comparison (Monte Carlo, 14-day detection delay)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.3,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scenario_comparison.png"), dpi=150)
    plt.close()


def plot_infection_curves_overlay(scenario_results_list):
    # all 6 infection curves on one plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (name, results) in enumerate(scenario_results_list):
        mean_I = results["mean"][:, :, I].sum(axis=1)
        lower_I = results["ci_lower"][:, :, I].sum(axis=1)
        upper_I = results["ci_upper"][:, :, I].sum(axis=1)
        days = np.arange(len(mean_I))
        ax.plot(days, mean_I, label=name, color=colors[idx % len(colors)], linewidth=2)
        ax.fill_between(days, lower_I, upper_I, alpha=0.1, color=colors[idx % len(colors)])

    # detection day line
    ax.axvline(x=DEFAULT_DELAY, color="black", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.text(DEFAULT_DELAY + 1.5, ax.get_ylim()[1] * 0.92,
            f"Virus detected\n(day {DEFAULT_DELAY})", fontsize=9, va="top")

    ax.set_xlabel("Days")
    ax.set_ylabel("Total Active Infections")
    ax.set_title("Active Infection Curves by Intervention Scenario (14-day detection delay)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "infection_curves_overlay.png"), dpi=150)
    plt.close()


def plot_group_breakdown(scenario_results_list):
    # infections per age group per scenario
    num_scenarios = len(scenario_results_list)
    bar_width = 0.25
    x = np.arange(num_scenarios)

    fig, ax = plt.subplots(figsize=(12, 6))
    for g in range(len(GROUP_NAMES)):
        means = [np.mean(r["total_infected_per_group"][:, g]) for _, r in scenario_results_list]
        stds = [np.std(r["total_infected_per_group"][:, g]) for _, r in scenario_results_list]
        offset = (g - 1) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                      label=GROUP_NAMES[g], color=GROUP_COLORS[g], edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, means):
            if val > 1000:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.2,
                        f"{val:,.0f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in scenario_results_list], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total Infections per Group")
    ax.set_title("Infection Breakdown by Demographic Group", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "group_breakdown.png"), dpi=150)
    plt.close()


def plot_infection_timeline(scenario_results_list):
    # timeline bars showing start peak end per group
    num_groups = len(GROUP_NAMES)
    fig, ax = plt.subplots(figsize=(11, 5))
    y_labels, y_pos = [], []
    pos = 0

    for scenario_name, results in scenario_results_list:
        for g in range(num_groups):
            start = np.mean(results["infection_start_day_per_group"][:, g])
            peak = np.mean(results["peak_day_per_group"][:, g])
            end = np.mean(results["infection_end_day_per_group"][:, g])

            ax.barh(pos, end - start, left=start, height=0.6,
                    color=GROUP_COLORS[g], alpha=0.5, edgecolor="black", linewidth=0.3)
            ax.plot(peak, pos, marker="v", color="black", markersize=8, zorder=5)
            ax.text(end + 1, pos, f"peak: day {peak:.0f}", va="center", fontsize=7)

            y_labels.append(GROUP_NAMES[g])
            y_pos.append(pos)
            pos += 1
        pos += 0.5

    # scenario labels
    pos = 0
    for scenario_name, _ in scenario_results_list:
        ax.text(-8, pos + (num_groups - 1)/2, scenario_name,
                va="center", ha="right", fontsize=8, fontweight="bold")
        pos += num_groups + 0.5

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Day")
    ax.set_title("Infection Timeline per Group (start to end, peak marked with triangle)", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(x=DEFAULT_DELAY, color="black", linestyle="--", alpha=0.5)
    ax.text(DEFAULT_DELAY + 0.5, -0.8, f"Detection\n(day {DEFAULT_DELAY})", fontsize=7, va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "infection_timeline.png"), dpi=150)
    plt.close()


def print_group_detail_table(scenario_results_list):
    # detailed per group stats
    for scenario_name, results in scenario_results_list:
        pop = results["params"]["population"]
        print(f"\n--- {scenario_name} ---")
        print(f"  {'Group':<25} {'Infected':>12} {'% of Group':>11} {'Peak Active':>12} {'Peak Day':>9} {'Start':>7} {'End':>7} {'Duration':>9}")
        print(f"  {'-'*93}")
        for g in range(len(GROUP_NAMES)):
            infected = np.mean(results["total_infected_per_group"][:, g])
            peak_active = np.mean(results["peak_infected_per_group"][:, g])
            peak_day = np.mean(results["peak_day_per_group"][:, g])
            start_day = np.mean(results["infection_start_day_per_group"][:, g])
            end_day = np.mean(results["infection_end_day_per_group"][:, g])
            print(f"  {GROUP_NAMES[g]:<25} {infected:>12,.0f} {infected/pop[g]*100:>10.1f}% "
                  f"{peak_active:>12,.0f} {peak_day:>9.1f} {start_day:>7.1f} {end_day:>7.1f} {end_day-start_day:>8.1f}d")
        total = np.mean(results["total_infected_per_run"])
        print(f"  {'TOTAL':<25} {total:>12,.0f} {total/pop.sum()*100:>10.1f}%")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_runs = 300
    print(f"Running Monte Carlo simulation with {num_runs} runs per scenario...")
    print(f"Interventions kick in after {DEFAULT_DELAY}-day detection delay\n")

    # run all 6 scenarios
    scenarios = get_all_scenarios()
    scenario_results_list = []
    for scenario_fn in scenarios:
        params, name = scenario_fn()
        print(f"  Simulating: {name}...", end=" ", flush=True)
        results = run_monte_carlo(params, num_runs=num_runs)
        scenario_results_list.append((name, results))
        print(f"done (mean total infected: {np.mean(results['total_infected_per_run']):,.0f})")

    # plots
    print("\nGenerating plots...")
    for name, results in scenario_results_list:
        plot_single_scenario(results, name)
    print("  - SEIR curves for all scenarios saved")

    plot_scenario_comparison(scenario_results_list)
    print("  - Scenario comparison bar chart saved")

    plot_infection_curves_overlay(scenario_results_list)
    print("  - Infection curves overlay saved")

    plot_group_breakdown(scenario_results_list)
    print("  - Group breakdown bar chart saved")

    plot_infection_timeline(scenario_results_list)
    print("  - Infection timeline chart saved")

    # summary table
    comparison = compute_scenario_comparison(scenario_results_list)
    print("\n" + "=" * 85)
    print(f"{'Scenario':<35} {'Mean Infected':>15} {'Attack Rate':>12} {'Peak Day':>10}")
    print("=" * 85)
    for name, stats in comparison.items():
        print(f"{name:<35} {stats['mean_total_infected']:>15,.0f} "
              f"{stats['mean_attack_rate']:>11.1%} {stats['mean_peak_day']:>10.1f}")
    print("=" * 85)

    print_group_detail_table(scenario_results_list)

    # extra analysis
    from analysis import run_all_analysis
    run_all_analysis(scenario_results_list)

    # validation
    from validation import run_validation
    run_validation(scenario_results_list)

    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
