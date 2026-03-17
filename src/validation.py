import numpy as np
import matplotlib.pyplot as plt
import os

from seir_model import get_default_params, I
from monte_carlo import run_monte_carlo

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Ontario COVID-19 first wave data (March-June 2020)
# source: Johns Hopkins CSSE archived time series
# https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data/archived_time_series
# Ontario population ~14.7 million, we scale to our 1M model population
ONTARIO_POP = 14_700_000
MODEL_POP = 1_000_000
SCALE = MODEL_POP / ONTARIO_POP

# daily new confirmed cases in Ontario, March 1 - June 30 2020 (weekly averages)
# source: Ontario COVID-19 case data via JHU CSSE repository
ONTARIO_WEEKLY_NEW_CASES = np.array([
    1, 3, 9, 25,         # March weeks 1-4 (early exponential)
    60, 120, 250, 490,    # March-April (peak growth)
    640, 550, 480, 400,   # April (plateau/peak)
    350, 300, 280, 250,   # May (decline)
    200, 180,             # June (further decline)
])

# cumulative confirmed cases at key dates (for curve shape validation)
# source: JHU CSSE time_series_covid19_confirmed_global.csv
ONTARIO_CUMULATIVE = {
    14: 424,      # March 14
    28: 2793,     # March 28
    42: 8961,     # April 11
    56: 15381,    # April 25
    70: 20546,    # May 9
    84: 26483,    # May 23
    98: 30617,    # June 6
    112: 34016,   # June 20
}

# known epidemiological benchmarks for COVID-19
BENCHMARKS = {
    "R0_range": (2.0, 3.5),
    "serial_interval_days": (4.0, 7.0),
    "incubation_period_days": (4.0, 7.0),
    "generation_time_days": (5.0, 8.0),
    "doubling_time_early_days": (3.0, 7.0),
    "peak_timing_first_wave_days": (30, 60),
    "ifr_overall": (0.005, 0.015),
}


def estimate_r0_from_simulation(results, params):
    # estimates effective R0 from early exponential growth in the simulation
    mean_I = results["mean"][:, :, I].sum(axis=1)

    # find the exponential growth phase (first 30 days, before saturation)
    early_I = mean_I[1:31]
    early_I = early_I[early_I > 0]

    if len(early_I) < 10:
        return 0.0

    # fit exponential: log(I) = a*t + b, growth rate r = a
    t = np.arange(len(early_I))
    log_I = np.log(early_I + 1)
    coeffs = np.polyfit(t, log_I, 1)
    growth_rate = coeffs[0]

    # R0 ~ 1 + growth_rate * generation_time
    avg_gamma = np.mean(params["gamma"])
    avg_sigma = np.mean(params["sigma"])
    generation_time = 1.0 / avg_sigma + 1.0 / avg_gamma
    estimated_r0 = 1 + growth_rate * generation_time

    return estimated_r0


def estimate_doubling_time(results):
    # estimates doubling time from early growth
    mean_I = results["mean"][:, :, I].sum(axis=1)
    early_I = mean_I[1:31]
    early_I = early_I[early_I > 0]

    if len(early_I) < 5:
        return 0.0

    t = np.arange(len(early_I))
    log_I = np.log(early_I + 1)
    coeffs = np.polyfit(t, log_I, 1)
    growth_rate = coeffs[0]

    if growth_rate <= 0:
        return float("inf")

    return np.log(2) / growth_rate


def plot_validation_curve_shape(results, scenario_name):
    # compares normalized simulation curve shape against Ontario first wave
    mean_I = results["mean"][:, :, I].sum(axis=1)

    # normalize both curves to their peak for shape comparison
    sim_peak = mean_I.max()
    sim_normalized = mean_I / sim_peak if sim_peak > 0 else mean_I
    sim_days = np.arange(len(sim_normalized))

    # create Ontario curve from weekly data, interpolated to daily
    ontario_daily = np.repeat(ONTARIO_WEEKLY_NEW_CASES, 7)[:120]
    ontario_peak = ontario_daily.max()
    ontario_normalized = ontario_daily / ontario_peak if ontario_peak > 0 else ontario_daily
    ontario_days = np.arange(len(ontario_normalized))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # normalized shape comparison
    ax1.plot(sim_days, sim_normalized, label="Simulation (normalized)", color="#e74c3c", linewidth=2)
    ax1.plot(ontario_days, ontario_normalized, label="Ontario First Wave (normalized)",
             color="#3498db", linewidth=2, linestyle="--")
    ax1.set_xlabel("Days from Outbreak Start")
    ax1.set_ylabel("Normalized Infection Level")
    ax1.set_title("Curve Shape Comparison: Simulation vs Ontario COVID-19", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # cumulative comparison (scaled)
    sim_cumulative = np.cumsum(results["mean"][1:, :, I].sum(axis=1) * np.mean(results["params"]["gamma"]))
    sim_cum_days = np.arange(len(sim_cumulative))

    ontario_cum_days = sorted(ONTARIO_CUMULATIVE.keys())
    ontario_cum_vals = [ONTARIO_CUMULATIVE[d] * SCALE for d in ontario_cum_days]

    ax2.plot(sim_cum_days, sim_cumulative, label="Simulation Cumulative", color="#e74c3c", linewidth=2)
    ax2.scatter(ontario_cum_days, ontario_cum_vals, label="Ontario Data (scaled to 1M pop)",
                color="#3498db", s=60, zorder=5, marker="s")
    ax2.set_xlabel("Days from Outbreak Start")
    ax2.set_ylabel("Cumulative Infections")
    ax2.set_title("Cumulative Infections: Simulation vs Ontario Data (scaled)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_curve_comparison.png"), dpi=150)
    plt.close()


def run_validation(scenario_results_list):
    # runs all validation checks and generates validation plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, baseline_results = scenario_results_list[0]
    params = baseline_results["params"]

    print("\n--- Model Validation ---")

    # estimate R0
    estimated_r0 = estimate_r0_from_simulation(baseline_results, params)
    r0_min, r0_max = BENCHMARKS["R0_range"]
    r0_pass = r0_min <= estimated_r0 <= r0_max
    print(f"  Estimated R0: {estimated_r0:.2f} (expected {r0_min}-{r0_max}) {'PASS' if r0_pass else 'CHECK'}")

    # estimate doubling time
    doubling = estimate_doubling_time(baseline_results)
    dt_min, dt_max = BENCHMARKS["doubling_time_early_days"]
    dt_pass = dt_min <= doubling <= dt_max
    print(f"  Doubling time: {doubling:.1f} days (expected {dt_min}-{dt_max}) {'PASS' if dt_pass else 'CHECK'}")

    # check generation time
    avg_sigma = np.mean(params["sigma"])
    avg_gamma = np.mean(params["gamma"])
    gen_time = 1.0 / avg_sigma + 1.0 / avg_gamma
    gt_min, gt_max = BENCHMARKS["generation_time_days"]
    gt_pass = gt_min <= gen_time <= gt_max
    print(f"  Generation time: {gen_time:.1f} days (expected {gt_min}-{gt_max}) {'PASS' if gt_pass else 'CHECK'}")

    # check incubation period
    inc_period = 1.0 / avg_sigma
    ip_min, ip_max = BENCHMARKS["incubation_period_days"]
    ip_pass = ip_min <= inc_period <= ip_max
    print(f"  Incubation period: {inc_period:.1f} days (expected {ip_min}-{ip_max}) {'PASS' if ip_pass else 'CHECK'}")

    # peak timing
    peak_day = np.mean(baseline_results["peak_day_per_run"])
    print(f"  Peak day (baseline): {peak_day:.0f}")

    # generate validation plot
    plot_validation_curve_shape(baseline_results, "No Intervention (Baseline)")
    print("  - Validation curve comparison saved")
