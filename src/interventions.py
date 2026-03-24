import numpy as np
from seir_model import get_default_params

# 14 day delay before any intervention kicks in
DEFAULT_DELAY = 14


def _apply_delay(params, start_day=DEFAULT_DELAY):
    # saves baseline contacts and sets when intervention starts
    baseline = get_default_params()
    params["baseline_contact_matrix"] = baseline["contact_matrix"].copy()
    params["intervention_start_day"] = start_day
    return params


def no_intervention():
    params = get_default_params()
    return params, "No Intervention (Baseline)"


def school_closure(start_day=DEFAULT_DELAY):
    # 70% reduction in children contacts
    params = get_default_params()
    params["contact_matrix"][0, :] *= 0.3
    params["contact_matrix"][:, 0] *= 0.3
    # undo double hit on diagonal, reapply once
    params["contact_matrix"][0, 0] /= 0.3
    params["contact_matrix"][0, 0] *= 0.3
    _apply_delay(params, start_day)
    return params, f"School Closures (day {start_day})"


def workplace_restriction(start_day=DEFAULT_DELAY):
    # 50% reduction in adult contacts
    params = get_default_params()
    params["contact_matrix"][1, :] *= 0.5
    params["contact_matrix"][:, 1] *= 0.5
    params["contact_matrix"][1, 1] /= 0.5
    params["contact_matrix"][1, 1] *= 0.5
    _apply_delay(params, start_day)
    return params, f"Workplace Restrictions (day {start_day})"


def elderly_isolation(start_day=DEFAULT_DELAY):
    # 60% reduction in elderly contacts
    params = get_default_params()
    params["contact_matrix"][2, :] *= 0.4
    params["contact_matrix"][:, 2] *= 0.4
    params["contact_matrix"][2, 2] /= 0.4
    params["contact_matrix"][2, 2] *= 0.4
    _apply_delay(params, start_day)
    return params, f"Elderly Isolation (day {start_day})"


def combined_moderate(start_day=DEFAULT_DELAY):
    # schools -50%, workplaces -30%, elderly -40% all at once
    params = get_default_params()
    cm = params["contact_matrix"].copy()

    # apply each reduction to rows and cols, fix diagonal double-hit
    for idx, factor in [(0, 0.5), (1, 0.7), (2, 0.6)]:
        cm[idx, :] *= factor
        cm[:, idx] *= factor
        cm[idx, idx] /= factor
        cm[idx, idx] *= factor

    params["contact_matrix"] = cm
    _apply_delay(params, start_day)
    return params, f"Combined Moderate (day {start_day})"


def full_lockdown(start_day=DEFAULT_DELAY):
    # 75% reduction across everything
    params = get_default_params()
    params["contact_matrix"] *= 0.25
    _apply_delay(params, start_day)
    return params, f"Full Lockdown (day {start_day})"


def get_all_scenarios():
    return [no_intervention, school_closure, workplace_restriction,
            elderly_isolation, combined_moderate, full_lockdown]
