"""
VarLend entry point — runs three benchmark scenarios and a baseline loan sweep.

``data/worker_profile.json`` is intentionally ``{}`` for teammates; this script uses an
in-memory ``WorkerProfile`` so the pipeline runs end-to-end without editing that file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.scenario_parser import parse_ai_scenario
from src.engine.monte_carlo import load_and_prepare, run_simulation, sweep_loan_space
from src.types import (
    AIScenario,
    DecayType,
    DiscreteJump,
    GigStream,
    GigType,
    LoanConfig,
    ParameterShift,
    ShiftTarget,
    ShiftType,
    SimulationConfig,
    WorkerProfile,
)


def _demo_profile() -> WorkerProfile:
    streams = [
        GigStream(
            platform_name="DoorDash",
            gig_type=GigType.DELIVERY,
            mean_monthly_income=4200.0,
            income_variance=45_000.0,
            tenure_months=22,
            is_primary=True,
        ),
        GigStream(
            platform_name="Uber",
            gig_type=GigType.RIDESHARE,
            mean_monthly_income=2800.0,
            income_variance=28_000.0,
            tenure_months=14,
            is_primary=False,
        ),
    ]
    return WorkerProfile(
        streams=streams,
        metro_area="Austin, TX",
        months_as_gig_worker=30,
        has_vehicle=True,
        has_dependents=False,
        liquid_savings=7500.0,
        monthly_fixed_expenses=2100.0,
        existing_debt_obligations=350.0,
        credit_score_range=(595, 635),
        loan_request_amount=6500.0,
        requested_term_months=36,
        acceptable_rate_range=(0.10, 0.20),
        correlation_matrix=None,
    )


def _mid_rate(profile: WorkerProfile) -> float:
    lo, hi = profile.acceptable_rate_range
    return 0.5 * (lo + hi)


def _print_row(label: str, res) -> None:
    rec = res.recommended_loan
    print(
        f"{label:16} | P(def)={res.p_default:6.3f} | EL={res.expected_loss:8.2f} | "
        f"CVaR95={res.cvar_95:8.2f} | Amt={rec.optimal_amount:8.2f} | Tier={rec.risk_tier.value}"
    )


def main() -> None:
    profile = _demo_profile()
    base_cfg = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=42)
    load = load_and_prepare(profile, base_cfg)
    rate = _mid_rate(profile)
    base_loan = LoanConfig(
        amount=profile.loan_request_amount,
        term_months=profile.requested_term_months,
        annual_rate=rate,
    )

    print("=== VarLend Monte Carlo benchmark (in-memory profile; worker_profile.json is {}) ===\n")

    baseline = run_simulation(profile, base_cfg, base_loan, load, None)
    _print_row("Baseline", baseline)

    recession_raw = {
        "narrative": "Macro slowdown with persistent income drag and higher jump risk.",
        "parameter_shifts": [
            {
                "target": "mu_base",
                "type": "multiplicative",
                "magnitude": 0.75,
                "start_month": 6,
                "duration_months": 18,
                "decay": "linear",
            },
            {
                "target": "lambda",
                "type": "multiplicative",
                "magnitude": 1.6,
                "start_month": 6,
                "duration_months": 18,
                "decay": "linear",
            },
            {
                "target": "sigma_base",
                "type": "multiplicative",
                "magnitude": 1.3,
                "start_month": 6,
                "duration_months": 18,
                "decay": "linear",
            },
        ],
        "discrete_jumps": [],
    }
    recession = parse_ai_scenario(recession_raw, base_cfg.horizon_months)
    recession_res = run_simulation(profile, base_cfg, base_loan, load, recession)
    _print_row("Recession", recession_res)

    injury = AIScenario(
        parameter_shifts=[
            ParameterShift(
                ShiftTarget.MU_BASE,
                ShiftType.MULTIPLICATIVE,
                0.35,
                4,
                2,
                DecayType.SNAP_BACK,
            )
        ],
        discrete_jumps=[DiscreteJump(4, -1800.0, 400.0)],
        narrative="Short injury shock with one-time medical outlay.",
    )
    injury_res = run_simulation(profile, base_cfg, base_loan, load, injury)
    _print_row("Injury", injury_res)

    print("\n=== Baseline sweep (max principal with P(default) < 0.08) ===\n")
    grid, optimal = sweep_loan_space(profile, base_cfg, load, None)
    print(f"Grid evaluations: {len(grid)}")
    if optimal is None:
        print("No configuration met P(default) < 0.08 on the grid.")
    else:
        print(
            f"Optimal: amount={optimal['amount']:.2f}, term={optimal['term_months']}m, "
            f"rate={optimal['annual_rate']:.2%}, P(def)={optimal['p_default']:.4f}, "
            f"EL={optimal['expected_loss']:.2f}, CVaR95={optimal['cvar_95']:.2f}"
        )


if __name__ == "__main__":
    main()
