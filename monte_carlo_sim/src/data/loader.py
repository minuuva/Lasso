"""
Worker profile ingestion from JSON plus derived portfolio parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.engine.correlation import effective_portfolio_mu_sigma
from src.types import (
    CorrelationMode,
    GigStream,
    GigType,
    JumpParams,
    LoadResult,
    SimulationConfig,
    WorkerProfile,
)


def _require_keys(d: dict[str, Any], keys: set[str], ctx: str) -> None:
    missing = keys - set(d.keys())
    if missing:
        raise ValueError(f"{ctx}: missing keys {sorted(missing)}")


def _parse_stream(obj: dict[str, Any], idx: int) -> GigStream:
    req = {
        "platform_name",
        "gig_type",
        "mean_monthly_income",
        "income_variance",
        "tenure_months",
        "is_primary",
    }
    _require_keys(obj, req, f"streams[{idx}]")
    try:
        gt = GigType(obj["gig_type"])
    except ValueError as e:
        raise ValueError(f"streams[{idx}].gig_type invalid") from e
    return GigStream(
        platform_name=str(obj["platform_name"]),
        gig_type=gt,
        mean_monthly_income=float(obj["mean_monthly_income"]),
        income_variance=float(obj["income_variance"]),
        tenure_months=int(obj["tenure_months"]),
        is_primary=bool(obj["is_primary"]),
    )


def load_worker_profile(
    path: str | Path,
    simulation_config: SimulationConfig | None = None,
) -> LoadResult:
    """
    Read ``worker_profile.json``, validate, and compute derived quantities.

    Parameters
    ----------
    path:
        Path to JSON file (typically ``data/worker_profile.json``).
    simulation_config:
        Used for correlation mode when collapsing streams; defaults to ``SimulationConfig()``.

    Returns
    -------
    LoadResult
        Profile, optional jump overrides, total monthly obligations, and portfolio (mu, sigma).
    """
    cfg = simulation_config or SimulationConfig()
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("worker profile file is empty — expected JSON object")
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError("worker profile root must be a JSON object")
    if data == {}:
        raise ValueError("worker profile is empty {} — populate fields per WorkerProfile schema")

    req_top = {
        "streams",
        "metro_area",
        "months_as_gig_worker",
        "has_vehicle",
        "has_dependents",
        "liquid_savings",
        "monthly_fixed_expenses",
        "existing_debt_obligations",
        "credit_score_range",
        "loan_request_amount",
        "requested_term_months",
        "acceptable_rate_range",
    }
    _require_keys(data, req_top, "worker profile")

    streams_raw = data["streams"]
    if not isinstance(streams_raw, list) or len(streams_raw) == 0:
        raise ValueError("streams must be a non-empty list")
    streams = [_parse_stream(s, i) for i, s in enumerate(streams_raw)]

    cr_mat = data.get("correlation_matrix")
    if cr_mat is not None:
        if not isinstance(cr_mat, list):
            raise ValueError("correlation_matrix must be a list of rows")
        cr_mat = [[float(x) for x in row] for row in cr_mat]

    profile = WorkerProfile(
        streams=streams,
        metro_area=str(data["metro_area"]),
        months_as_gig_worker=int(data["months_as_gig_worker"]),
        has_vehicle=bool(data["has_vehicle"]),
        has_dependents=bool(data["has_dependents"]),
        liquid_savings=float(data["liquid_savings"]),
        monthly_fixed_expenses=float(data["monthly_fixed_expenses"]),
        existing_debt_obligations=float(data["existing_debt_obligations"]),
        credit_score_range=(int(data["credit_score_range"][0]), int(data["credit_score_range"][1])),
        loan_request_amount=float(data["loan_request_amount"]),
        requested_term_months=int(data["requested_term_months"]),
        acceptable_rate_range=(float(data["acceptable_rate_range"][0]), float(data["acceptable_rate_range"][1])),
        correlation_matrix=cr_mat,
    )

    if cfg.correlation_mode == CorrelationMode.CUSTOM_MATRIX and profile.correlation_matrix is None:
        raise ValueError("correlation_matrix required when correlation_mode is CUSTOM_MATRIX")

    jp = JumpParams(
        jump_lambda=float(data["lambda"]) if "lambda" in data else None,
        mu_jump=float(data["mu_jump"]) if "mu_jump" in data else None,
        sigma_jump=float(data["sigma_jump"]) if "sigma_jump" in data else None,
    )

    total_obligations = profile.monthly_fixed_expenses + profile.existing_debt_obligations
    mu_eff, sig_eff = effective_portfolio_mu_sigma(profile, cfg)

    return LoadResult(
        profile=profile,
        jump_params=jp,
        total_monthly_obligations=total_obligations,
        effective_mu_base=mu_eff,
        effective_sigma_base=sig_eff,
    )
