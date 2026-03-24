import numpy as np

# group indices
CHILDREN, ADULTS, ELDERLY = 0, 1, 2
GROUP_NAMES = ["Children (0-17)", "Working Adults (18-64)", "Elderly (65+)"]

# compartment indices
S, E, I, R = 0, 1, 2, 3


def get_default_params():
    # default disease params
    params = {
        "population": np.array([200_000, 600_000, 200_000]),  # 1M total
        "initial_infected": np.array([5, 15, 5]),
        "initial_exposed": np.array([10, 30, 10]),
        "sigma": np.array([1/5.2, 1/5.2, 1/5.2]),     # incubation rate
        "gamma": np.array([1/7.0, 1/10.0, 1/12.0]),    # recovery rate
        "beta": np.array([0.03, 0.035, 0.04]),          # transmission rate
        # contact matrix from POLYMOD
        "contact_matrix": np.array([
            [10.0,  3.0, 1.5],
            [ 3.0,  8.0, 2.0],
            [ 1.5,  2.0, 4.0],
        ]),
        "days": 120,
        "dt": 1.0,
        "noise_scale": 0.05,  # 5% noise for mc
        "intervention_start_day": 0,
        "baseline_contact_matrix": None,
    }
    return params


def initialize_state(params):
    # starting S E I R for each group
    num_groups = len(params["population"])
    state = np.zeros((num_groups, 4))
    for g in range(num_groups):
        state[g, I] = params["initial_infected"][g]
        state[g, E] = params["initial_exposed"][g]
        state[g, S] = params["population"][g] - params["initial_infected"][g] - params["initial_exposed"][g]
    return state


def step_seir(state, params, rng=None):
    # one day step
    if rng is None:
        rng = np.random.default_rng()

    dt = params["dt"]
    num_groups = len(params["population"])
    new_state = state.copy()

    for g in range(num_groups):
        N_g = params["population"][g]
        if N_g == 0:
            continue
        s, e, i, r = state[g]

        # force of infection from all groups
        foi = 0.0
        for j in range(num_groups):
            N_j = params["population"][j]
            if N_j == 0:
                continue
            foi += params["beta"][g] * params["contact_matrix"][j, g] * state[j, I] / N_j

        # stochastic noise
        noise = max(1.0 + rng.normal(0, params["noise_scale"]), 0.0)
        foi *= noise

        # transitions
        new_exposed = min(foi * s * dt, s)
        new_infected = min(params["sigma"][g] * e * dt, e)
        new_recovered = min(params["gamma"][g] * i * dt, i)

        # update compartments
        new_state[g, S] = s - new_exposed
        new_state[g, E] = e + new_exposed - new_infected
        new_state[g, I] = i + new_infected - new_recovered
        new_state[g, R] = r + new_recovered
        new_state[g] = np.maximum(new_state[g], 0.0)

    return new_state


def run_simulation(params, seed=None):
    # full 120 day sim w detection delay
    rng = np.random.default_rng(seed)
    num_steps = int(params["days"] / params["dt"])
    num_groups = len(params["population"])

    # save both contact matrices for delay swap
    intervention_cm = params["contact_matrix"].copy()
    baseline_cm = params.get("baseline_contact_matrix")
    start_day = params.get("intervention_start_day", 0)
    has_delay = baseline_cm is not None and start_day > 0

    history = np.zeros((num_steps + 1, num_groups, 4))
    state = initialize_state(params)
    history[0] = state.copy()

    for t in range(1, num_steps + 1):
        # baseline before detection, intervention after
        if has_delay and t < start_day:
            params["contact_matrix"] = baseline_cm
        else:
            params["contact_matrix"] = intervention_cm

        state = step_seir(state, params, rng)
        history[t] = state.copy()

    params["contact_matrix"] = intervention_cm
    return history
