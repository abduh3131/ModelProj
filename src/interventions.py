import numpy as np
from seir_model import get_default_params

# default delay: virus spreads undetected for 14 days before response
DEFAULT_DELAY = 14


def _apply_delay(params, start_day=DEFAULT_DELAY):
    # saves the baseline (unmodified) contact matrix and sets the intervention start day
    baseline = get_default_params()
    params["baseline_contact_matrix"] = baseline["contact_matrix"].copy()
    params["intervention_start_day"] = start_day
    return params


def no_intervention():
    # baseline with no changes (no delay needed since there's nothing to delay)
    params = get_default_params()
    return params, "No Intervention (Baseline)"


def school_closure(start_day=DEFAULT_DELAY):
    # reduces children contacts by 70%, kicks in after start_day
    params = get_default_params()
    params["contact_matrix"][0, :] *= 0.3
    params["contact_matrix"][:, 0] *= 0.3
    params["contact_matrix"][0, 0] /= 0.3
    params["contact_matrix"][0, 0] *= 0.3
    _apply_delay(params, start_day)
    return params, f"School Closures (day {start_day})"


def workplace_restriction(start_day=DEFAULT_DELAY):
    # reduces adult contacts by 50%
    params = get_default_params()
    params["contact_matrix"][1, :] *= 0.5
    params["contact_matrix"][:, 1] *= 0.5
    params["contact_matrix"][1, 1] /= 0.5
    params["contact_matrix"][1, 1] *= 0.5
    _apply_delay(params, start_day)
    return params, f"Workplace Restrictions (day {start_day})"


def elderly_isolation(start_day=DEFAULT_DELAY):
    # reduces elderly contacts by 60%
    params = get_default_params()
    params["contact_matrix"][2, :] *= 0.4
    params["contact_matrix"][:, 2] *= 0.4
    params["contact_matrix"][2, 2] /= 0.4
    params["contact_matrix"][2, 2] *= 0.4
    _apply_delay(params, start_day)
    return params, f"Elderly Isolation (day {start_day})"


def combined_moderate(start_day=DEFAULT_DELAY):
    # schools -50%, workplaces -30%, elderly -40%
    params = get_default_params()
    cm = params["contact_matrix"].copy()

    school_factor = 0.5
    cm[0, :] *= school_factor
    cm[:, 0] *= school_factor
    cm[0, 0] /= school_factor
    cm[0, 0] *= school_factor

    work_factor = 0.7
    cm[1, :] *= work_factor
    cm[:, 1] *= work_factor
    cm[1, 1] /= work_factor
    cm[1, 1] *= work_factor

    elderly_factor = 0.6
    cm[2, :] *= elderly_factor
    cm[:, 2] *= elderly_factor
    cm[2, 2] /= elderly_factor
    cm[2, 2] *= elderly_factor

    params["contact_matrix"] = cm
    _apply_delay(params, start_day)
    return params, f"Combined Moderate (day {start_day})"


def full_lockdown(start_day=DEFAULT_DELAY):
    # all contacts reduced by 75%
    params = get_default_params()
    params["contact_matrix"] *= 0.25
    _apply_delay(params, start_day)
    return params, f"Full Lockdown (day {start_day})"


def get_all_scenarios():
    # returns list of all scenario functions (called with default delay)
    return [
        no_intervention,
        school_closure,
        workplace_restriction,
        elderly_isolation,
        combined_moderate,
        full_lockdown,
    ]
