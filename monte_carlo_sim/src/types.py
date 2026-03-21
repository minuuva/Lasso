"""
Central type definitions for VarLend — dataclasses and enums used across the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class GigType(Enum):
    DELIVERY = "delivery"
    RIDESHARE = "rideshare"
    FREELANCE = "freelance"
    MIXED = "mixed"


class MacroState(Enum):
    NORMAL = "normal"
    RECESSION = "recession"
    GAS_SPIKE = "gas_spike"
    CUSTOM = "custom"


class CorrelationMode(Enum):
    INDEPENDENT = "independent"
    CORRELATED = "correlated"
    CUSTOM_MATRIX = "custom_matrix"


class ShiftTarget(Enum):
    MU_BASE = "mu_base"
    SIGMA_BASE = "sigma_base"
    LAMBDA = "lambda"
    EXPENSES = "expenses"


class ShiftType(Enum):
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


class DecayType(Enum):
    SNAP_BACK = "snap_back"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class RiskTier(Enum):
    PRIME = "prime"
    NEAR_PRIME = "near_prime"
    SUBPRIME = "subprime"
    DECLINE = "decline"


@dataclass(frozen=True)
class GigStream:
    platform_name: str
    gig_type: GigType
    mean_monthly_income: float
    income_variance: float
    tenure_months: int
    is_primary: bool


@dataclass(frozen=True)
class WorkerProfile:
    streams: list[GigStream]
    metro_area: str
    months_as_gig_worker: int
    has_vehicle: bool
    has_dependents: bool
    liquid_savings: float
    monthly_fixed_expenses: float
    existing_debt_obligations: float
    credit_score_range: tuple[int, int]
    loan_request_amount: float
    requested_term_months: int
    acceptable_rate_range: tuple[float, float]
    correlation_matrix: Optional[list[list[float]]] = None


@dataclass
class SimulationConfig:
    n_paths: int = 5000
    horizon_months: int = 24
    random_seed: Optional[int] = None
    macro_state: MacroState = MacroState.NORMAL
    correlation_mode: CorrelationMode = CorrelationMode.CORRELATED


@dataclass(frozen=True)
class ParameterShift:
    target: ShiftTarget
    type: ShiftType
    magnitude: float
    start_month: int
    duration_months: int
    decay: DecayType


@dataclass(frozen=True)
class DiscreteJump:
    month: int
    amount: float
    variance: float
    echo_months: Optional[int] = None
    echo_decay_rate: Optional[float] = None


@dataclass
class AIScenario:
    parameter_shifts: list[ParameterShift]
    discrete_jumps: list[DiscreteJump]
    narrative: str = ""


@dataclass(frozen=True)
class LoanConfig:
    amount: float
    term_months: int
    annual_rate: float


@dataclass
class PathResult:
    monthly_incomes: np.ndarray
    monthly_net_cash_flows: np.ndarray
    monthly_buffer: np.ndarray
    defaulted: bool
    default_month: Optional[int]
    loss_given_default: float
    total_lender_profit: float


@dataclass
class LoanRecommendation:
    approved: bool
    optimal_amount: float
    optimal_term_months: int
    optimal_rate: float
    risk_tier: RiskTier
    reasoning: list[str]
    alternative_structures: list[dict[str, Any]]


@dataclass
class SimulationResult:
    p_default: float
    expected_loss: float
    cvar_95: float
    median_income_by_month: np.ndarray
    p10_income_by_month: np.ndarray
    p90_income_by_month: np.ndarray
    time_to_default_percentiles: dict[str, float]
    recommended_loan: LoanRecommendation
    raw_paths: np.ndarray


@dataclass(frozen=True)
class JumpParams:
    """Optional jump-process overrides loaded from worker profile JSON."""

    jump_lambda: Optional[float] = None
    mu_jump: Optional[float] = None
    sigma_jump: Optional[float] = None


@dataclass(frozen=True)
class LoadResult:
    profile: WorkerProfile
    jump_params: JumpParams
    total_monthly_obligations: float
    effective_mu_base: float
    effective_sigma_base: float
