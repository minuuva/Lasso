"""
Strict validation and parsing of AI-produced JSON-like scenario dictionaries.
"""

from __future__ import annotations

from typing import Any

from src.types import AIScenario, DecayType, DiscreteJump, ParameterShift, ShiftTarget, ShiftType


def parse_ai_scenario(raw: dict[str, Any], horizon_months: int) -> AIScenario:
    """
    Validate ``raw`` against the ``AIScenario`` schema and return a typed object.

    Raises
    ------
    ValueError
        On unknown fields, invalid enums, out-of-range magnitudes, or bad month indices.
    """
    allowed = {"parameter_shifts", "discrete_jumps", "narrative"}
    extras = set(raw.keys()) - allowed
    if extras:
        raise ValueError(f"Unknown AIScenario fields: {sorted(extras)}")

    narrative = str(raw.get("narrative", ""))
    shifts_raw = raw.get("parameter_shifts", [])
    jumps_raw = raw.get("discrete_jumps", [])

    if not isinstance(shifts_raw, list):
        raise ValueError("parameter_shifts must be a list")
    if not isinstance(jumps_raw, list):
        raise ValueError("discrete_jumps must be a list")

    shifts: list[ParameterShift] = []
    for i, item in enumerate(shifts_raw):
        if not isinstance(item, dict):
            raise ValueError(f"parameter_shifts[{i}] must be an object")
        req = {"target", "type", "magnitude", "start_month", "duration_months", "decay"}
        if set(item.keys()) != req:
            raise ValueError(f"parameter_shifts[{i}] must contain exactly {sorted(req)}")
        try:
            target = ShiftTarget(item["target"])
            stype = ShiftType(item["type"])
            decay = DecayType(item["decay"])
        except ValueError as e:
            raise ValueError(f"parameter_shifts[{i}] invalid enum: {e}") from e
        mag = float(item["magnitude"])
        if stype == ShiftType.MULTIPLICATIVE:
            mag = float(min(max(mag, 0.05), 3.0))
        start = int(item["start_month"])
        dur = int(item["duration_months"])
        if dur <= 0:
            raise ValueError(f"parameter_shifts[{i}].duration_months must be > 0")
        if start < 0 or start >= horizon_months:
            raise ValueError(f"parameter_shifts[{i}].start_month out of horizon")
        if start + dur > horizon_months:
            raise ValueError(f"parameter_shifts[{i}] extends past horizon_months={horizon_months}")
        shifts.append(ParameterShift(target, stype, mag, start, dur, decay))

    jumps: list[DiscreteJump] = []
    allowed_jump = {"month", "amount", "variance", "echo_months", "echo_decay_rate"}
    for j, item in enumerate(jumps_raw):
        if not isinstance(item, dict):
            raise ValueError(f"discrete_jumps[{j}] must be an object")
        keys = set(item.keys())
        if not keys <= allowed_jump:
            raise ValueError(f"discrete_jumps[{j}] contains unknown keys {sorted(keys - allowed_jump)}")
        if "month" not in keys or "amount" not in keys or "variance" not in keys:
            raise ValueError(f"discrete_jumps[{j}] must include month, amount, and variance")
        month = int(item["month"])
        if month < 0 or month >= horizon_months:
            raise ValueError(f"discrete_jumps[{j}].month out of horizon")
        amt = float(min(max(item["amount"], -50000.0), 50000.0))
        var = float(max(item["variance"], 0.0))
        em = item.get("echo_months")
        ed = item.get("echo_decay_rate")
        jumps.append(
            DiscreteJump(
                month,
                amt,
                var,
                int(em) if em is not None else None,
                float(ed) if ed is not None else None,
            )
        )

    return AIScenario(shifts, jumps, narrative)
