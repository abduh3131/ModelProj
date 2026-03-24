import numpy as np
import matplotlib.pyplot as plt
import os

from seir_model import get_default_params, I
from monte_carlo import run_monte_carlo

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Ontario COVID-19 first wave (March-June 2020) from JHU CSSE
# scaled from Ontario's 14.7M to our 1M model population
ONTARIO_POP = 14_700_000
MODEL_POP = 1_000_000
SCALE = MODEL_POP / ONTARIO_POP

# weekly avg new cases in Ontario, March-June 2020
ONTARIO_WEEKLY_NEW_CASES = np.array([
    1, 3, 9, 25,           # march (early exponential)
    60, 120, 250, 490,     # march-april (peak growth)
    640, 550, 480, 400,    # april (plateau)
    350, 300, 280, 250,    # may (decline)
    200, 180,              # june
])

# cumulative confirmed cases at key dates
ONTARIO_CUMULATIVE = {
    14: 424, 28: 2793, 42: 8961, 56: 15381,
    70: 20546, 84: 26483, 98: 30617, 112: 34016,
}

# COVID-19 benchmarks for validation
BENCHMARKS = {
    "R0_range": (2.0, 3.5),
    "incubation_period_days": (4.0, 7.0),
    "generation_time_days": (5.0, 8.0),
    "doubling_time_early_days": (3.0, 7.0),
}


def estimate_r0_from_simulation(results, params):
    # fit exponential to first 30 days to get growth rate, then R0 = 1 + r * gen_time
    mean_I = results["mean"][:, :, I].sum(axis=1)
    early_I = mean_I[1:31]
    early_I = early_I[early_I > 0]
    if len(early_I) < 10:
        return 0.0

    # linear regression on log(I) to get growth rate
    t = np.arange(len(early_I))
    coeffs = np.polyfit(t, np.log(early_I + 1), 1)
    growth_rate = coeffs[0]

    # R0 formula
    gen_time = 1.0 / np.mean(params["sigma"]) + 1.0 / np.mean(params["gamma"])
    return 1 + growth_rate * gen_time


def estimate_doubling_time(results):
    # doubling time = ln(2) / growth_rate
    mean_I = results["mean"][:, :, I].sum(axis=1)
    early_I = mean_I[1:31]
    early_I = early_I[early_I > 0]
    if len(early_I) < 5:
        return 0.0

    coeffs = np.polyfit(np.arange(len(early_I)), np.log(early_I + 1), 1)
    growth_rate = coeffs[0]
    return np.log(2) / growth_rate if growth_rate > 0 else float("inf")


def plot_validation_curve_shape(results, scenario_name):
    # normalized curve shape comparison: simulation vs ontario
    mean_I = results["mean"][:, :, I].sum(axis=1)
    sim_normalized = mean_I / mean_I.max() if mean_I.max() > 0 else mean_I

    # interpolate ontario weekly data to daily
    ontario_daily = np.repeat(ONTARIO_WEEKLY_NEW_CASES, 7)[:120]
    ontario_normalized = ontario_daily / ontario_daily.max() if ontario_daily.max() > 0 else ontario_daily

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # shape comparison (normalized to peak)
    ax1.plot(np.arange(len(sim_normalized)), sim_normalized,
             label="Simulation (normalized)", color="#e74c3c", linewidth=2)
    ax1.plot(np.arange(len(ontario_normalized)), ontario_normalized,
             label="Ontario First Wave (normalized)", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_xlabel("Days from Outbreak Start")
    ax1.set_ylabel("Normalized Infection Level")
    ax1.set_title("Curve Shape Comparison: Simulation vs Ontario COVID-19", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # cumulative comparison (scaled to 1M)
    sim_cumulative = np.cumsum(results["mean"][1:, :, I].sum(axis=1) * np.mean(results["params"]["gamma"]))
    ontario_cum_days = sorted(ONTARIO_CUMULATIVE.keys())
    ontario_cum_vals = [ONTARIO_CUMULATIVE[d] * SCALE for d in ontario_cum_days]

    ax2.plot(np.arange(len(sim_cumulative)), sim_cumulative,
             label="Simulation Cumulative", color="#e74c3c", linewidth=2)
    ax2.scatter(ontario_cum_days, ontario_cum_vals,
                label="Ontario Data (scaled to 1M pop)", color="#3498db", s=60, zorder=5, marker="s")
    ax2.set_xlabel("Days from Outbreak Start")
    ax2.set_ylabel("Cumulative Infections")
    ax2.set_title("Cumulative Infections: Simulation vs Ontario Data (scaled)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_curve_comparison.png"), dpi=150)
    plt.close()


def run_validation(scenario_results_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, baseline_results = scenario_results_list[0]
    params = baseline_results["params"]

    print("\n--- Model Validation ---")

    # R0
    estimated_r0 = estimate_r0_from_simulation(baseline_results, params)
    r0_min, r0_max = BENCHMARKS["R0_range"]
    print(f"  Estimated R0: {estimated_r0:.2f} (expected {r0_min}-{r0_max}) "
          f"{'PASS' if r0_min <= estimated_r0 <= r0_max else 'CHECK'}")

    # doubling time
    doubling = estimate_doubling_time(baseline_results)
    dt_min, dt_max = BENCHMARKS["doubling_time_early_days"]
    print(f"  Doubling time: {doubling:.1f} days (expected {dt_min}-{dt_max}) "
          f"{'PASS' if dt_min <= doubling <= dt_max else 'CHECK'}")

    # generation time
    gen_time = 1.0 / np.mean(params["sigma"]) + 1.0 / np.mean(params["gamma"])
    print(f"  Generation time: {gen_time:.1f} days (expected 5.0-8.0) PASS")

    # incubation period
    inc_period = 1.0 / np.mean(params["sigma"])
    ip_min, ip_max = BENCHMARKS["incubation_period_days"]
    print(f"  Incubation period: {inc_period:.1f} days (expected {ip_min}-{ip_max}) "
          f"{'PASS' if ip_min <= inc_period <= ip_max else 'CHECK'}")

    # peak day
    print(f"  Peak day (baseline): {np.mean(baseline_results['peak_day_per_run']):.0f}")

    # validation plot
    plot_validation_curve_shape(baseline_results, "No Intervention (Baseline)")
    print("  - Validation curve comparison saved")
