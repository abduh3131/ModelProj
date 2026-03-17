import numpy as np

# group indices
CHILDREN = 0
ADULTS = 1
ELDERLY = 2

GROUP_NAMES = ["Children (0-17)", "Working Adults (18-64)", "Elderly (65+)"]

# compartment indices
S, E, I, R = 0, 1, 2, 3
COMPARTMENT_NAMES = ["Susceptible", "Exposed", "Infected", "Recovered"]


def get_default_params():
    params = {
        # population sizes per group
        "population": np.array([200_000, 600_000, 200_000]),
        # initial infected per group
        "initial_infected": np.array([5, 15, 5]),
        # initial exposed per group
        "initial_exposed": np.array([10, 30, 10]),
        # incubation rate (1 / avg incubation days)
        "sigma": np.array([1 / 5.2, 1 / 5.2, 1 / 5.2]),
        # recovery rate (1 / avg infectious days)
        "gamma": np.array([1 / 7.0, 1 / 10.0, 1 / 12.0]),
        # base transmission probability per contact
        "beta": np.array([0.03, 0.035, 0.04]),
        # contact matrix: avg daily contacts between groups
        "contact_matrix": np.array([
            [10.0,  3.0,  1.5],
            [ 3.0,  8.0,  2.0],
            [ 1.5,  2.0,  4.0],
        ]),
        # simulation length in days
        "days": 120,
        # time step
        "dt": 1.0,
        # noise scale for stochastic variation
        "noise_scale": 0.05,
        # day the intervention kicks in (0 = immediate, 14 = 2 week delay)
        "intervention_start_day": 0,
        # baseline contact matrix used before intervention starts
        # (set automatically by intervention functions)
        "baseline_contact_matrix": None,
    }
    return params


def initialize_state(params):
    # sets up initial SEIR state for all groups
    num_groups = len(params["population"])
    state = np.zeros((num_groups, 4))

    for g in range(num_groups):
        state[g, I] = params["initial_infected"][g]
        state[g, E] = params["initial_exposed"][g]
        state[g, R] = 0
        state[g, S] = (
            params["population"][g]
            - params["initial_infected"][g]
            - params["initial_exposed"][g]
        )

    return state


def step_seir(state, params, rng=None):
    # advances SEIR model by one time step with stochastic noise
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

        # calculate force of infection from all groups
        force_of_infection = 0.0
        for j in range(num_groups):
            N_j = params["population"][j]
            if N_j == 0:
                continue
            force_of_infection += (
                params["beta"][g]
                * params["contact_matrix"][j, g]
                * state[j, I]
                / N_j
            )

        # add stochastic noise
        noise = 1.0 + rng.normal(0, params["noise_scale"])
        noise = max(noise, 0.0)
        force_of_infection *= noise

        # compute transitions
        new_exposed = force_of_infection * s * dt
        new_infected = params["sigma"][g] * e * dt
        new_recovered = params["gamma"][g] * i * dt

        # clamp transitions to available population
        new_exposed = min(new_exposed, s)
        new_infected = min(new_infected, e)
        new_recovered = min(new_recovered, i)

        # update compartments
        new_state[g, S] = s - new_exposed
        new_state[g, E] = e + new_exposed - new_infected
        new_state[g, I] = i + new_infected - new_recovered
        new_state[g, R] = r + new_recovered

        # clamp to non-negative
        new_state[g] = np.maximum(new_state[g], 0.0)

    return new_state


def run_simulation(params, seed=None):
    # runs a single SEIR simulation over the full time horizon
    rng = np.random.default_rng(seed)
    num_steps = int(params["days"] / params["dt"])
    num_groups = len(params["population"])

    # store the intervention contact matrix
    intervention_cm = params["contact_matrix"].copy()
    baseline_cm = params.get("baseline_contact_matrix")
    start_day = params.get("intervention_start_day", 0)

    # if no baseline was set, there's no delay to apply
    has_delay = baseline_cm is not None and start_day > 0

    history = np.zeros((num_steps + 1, num_groups, 4))
    state = initialize_state(params)
    history[0] = state.copy()

    for t in range(1, num_steps + 1):
        # use baseline matrix before intervention starts, intervention matrix after
        if has_delay and t < start_day:
            params["contact_matrix"] = baseline_cm
        else:
            params["contact_matrix"] = intervention_cm

        state = step_seir(state, params, rng)
        history[t] = state.copy()

    # restore original matrix
    params["contact_matrix"] = intervention_cm
    return history
