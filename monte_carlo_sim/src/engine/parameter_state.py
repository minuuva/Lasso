"""
Deterministic parameter shift schedule — scalar effective parameters per simulation month.
"""

from __future__ import annotations

from src.types import DecayType, MacroState, ParameterShift, ShiftTarget, ShiftType


def _shift_active(shift: ParameterShift, month: int) -> bool:
    end = shift.start_month + shift.duration_months
    return shift.start_month <= month < end


def _decayed_magnitude(shift: ParameterShift, month: int) -> float:
    if not _shift_active(shift, month):
        return 0.0
    months_since = month - shift.start_month
    if shift.decay == DecayType.SNAP_BACK:
        return float(shift.magnitude)
    if shift.decay == DecayType.LINEAR:
        if shift.duration_months <= 0:
            return 0.0
        return float(shift.magnitude) * (1.0 - months_since / shift.duration_months)
    if shift.decay == DecayType.EXPONENTIAL:
        return float(shift.magnitude) * (0.7**months_since)
    return float(shift.magnitude)


def effective_parameters(
    month: int,
    mu_base: float,
    sigma_base: float,
    lambda_base: float,
    expenses_base: float,
    shifts: list[ParameterShift],
) -> tuple[float, float, float, float]:
    """
    Apply all active ``ParameterShift`` objects for ``month`` to scalar bases.

    Multiplicative shifts compound as a product; additive shifts sum. Multiplicative
    adjustments are applied to the base first, then additive terms are applied.

    Returns
    -------
    tuple[float, float, float, float]
        (mu_base_eff, sigma_base_eff, lambda_eff, expenses_eff).
    """
    mult = {ShiftTarget.MU_BASE: 1.0, ShiftTarget.SIGMA_BASE: 1.0, ShiftTarget.LAMBDA: 1.0, ShiftTarget.EXPENSES: 1.0}
    add = {ShiftTarget.MU_BASE: 0.0, ShiftTarget.SIGMA_BASE: 0.0, ShiftTarget.LAMBDA: 0.0, ShiftTarget.EXPENSES: 0.0}

    for sh in shifts:
        if not _shift_active(sh, month):
            continue
        mag = _decayed_magnitude(sh, month)
        if sh.type == ShiftType.MULTIPLICATIVE:
            mult[sh.target] *= mag
        else:
            add[sh.target] += mag

    mu_eff = mu_base * mult[ShiftTarget.MU_BASE] + add[ShiftTarget.MU_BASE]
    sigma_eff = sigma_base * mult[ShiftTarget.SIGMA_BASE] + add[ShiftTarget.SIGMA_BASE]
    lambda_eff = lambda_base * mult[ShiftTarget.LAMBDA] + add[ShiftTarget.LAMBDA]
    exp_eff = expenses_base * mult[ShiftTarget.EXPENSES] + add[ShiftTarget.EXPENSES]
    return mu_eff, sigma_eff, lambda_eff, exp_eff


def macro_scaling(macro_state: MacroState | None) -> tuple[float, float, float, float]:
    """
    Optional coarse macro multipliers (mu, sigma, lambda, expenses).

    Used only for built-in macro labels; AI scenarios handle detailed shocks.
    """
    if macro_state is None:
        return 1.0, 1.0, 1.0, 1.0
    if macro_state == MacroState.NORMAL:
        return 1.0, 1.0, 1.0, 1.0
    if macro_state == MacroState.RECESSION:
        return 0.9, 1.15, 1.2, 1.05
    if macro_state == MacroState.GAS_SPIKE:
        return 0.95, 1.1, 1.05, 1.12
    return 1.0, 1.0, 1.0, 1.0
