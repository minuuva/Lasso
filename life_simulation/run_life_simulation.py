"""
Run Life Simulation - Integration entry point for Layer 1 + Layer 2.

Connects life trajectory generation with Monte Carlo execution.
"""

from typing import Optional, Any, Dict
import sys
import os

# Add parent directory and monte_carlo_sim to path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
monte_carlo_dir = os.path.join(parent_dir, 'monte_carlo_sim')

sys.path.insert(0, parent_dir)
sys.path.insert(0, monte_carlo_dir)

from data_pipeline.loaders import DataLoader

# Import Monte Carlo components
from src.types import (
    SimulationConfig,
    LoanConfig,
    SimulationResult,
    WorkerProfile,
    GigType,
    CorrelationMode
)
from src.integration.profile_builder import (
    build_profile_from_application,
    CustomerApplication
)
from src.engine.monte_carlo import run_simulation, load_and_prepare
from src.ai.scenario_parser import parse_ai_scenario


def run_full_life_simulation(
    archetype_id: str,
    customer_application: CustomerApplication,
    loan_config: LoanConfig,
    random_seed: Optional[int] = None,
    n_paths: int = 5000,
    horizon_months: int = 24,
    ai_scenario: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    """
    Run Monte Carlo simulation with per-path life event sampling.
    
    Each of the 5000 paths independently samples:
    - Life events (vehicle repairs, health issues, platform deactivations, housing)
    - Macro shocks (recession, gas spike, regulatory changes, tech disruption)
    - Event impacts on income and expenses
    
    This provides true path independence where each Monte Carlo path has its
    own unique life trajectory, resulting in realistic probability distributions.
    
    Args:
        archetype_id: Archetype ID from archetypes.json (e.g., "volatile_vic")
        customer_application: Customer loan application data
        loan_config: Loan parameters (amount, term, rate)
        random_seed: Optional seed for reproducibility (None for true randomness)
        n_paths: Number of Monte Carlo paths (default 5000)
        horizon_months: Simulation horizon in months (default 24)
        ai_scenario: Optional raw AIScenario dict (narrative, parameter_shifts,
            discrete_jumps) from AI / frontend; merged with path-level sampling.
    
    Returns:
        SimulationResult with realistic P(default) distribution
    """
    loader = DataLoader()
    profile = build_profile_from_application(customer_application, loader)
    archetype_data = loader.load_archetype(archetype_id)
    
    config = SimulationConfig(
        n_paths=n_paths,
        horizon_months=horizon_months,
        random_seed=random_seed,
        correlation_mode=CorrelationMode.CORRELATED
    )
    
    load = load_and_prepare(profile, config)

    parsed_scenario = None
    if ai_scenario:
        parsed_scenario = parse_ai_scenario(ai_scenario, horizon_months)
    
    # Pass archetype data for per-path event sampling; optional AIScenario stress overlay
    result = run_simulation(
        profile, 
        config, 
        loan_config, 
        load,
        scenario=parsed_scenario,
        archetype_data=archetype_data
    )
    
    return result


def run_static_simulation(
    customer_application: CustomerApplication,
    loan_config: LoanConfig,
    random_seed: Optional[int] = None,
    n_paths: int = 5000,
    horizon_months: int = 24
) -> SimulationResult:
    """
    Run Monte Carlo simulation WITHOUT life simulation (Layer 1 only).
    
    This is the baseline static simulation for comparison.
    
    Args:
        customer_application: Customer data
        loan_config: Loan parameters
        random_seed: Optional seed
        n_paths: Number of paths
        horizon_months: Simulation horizon
    
    Returns:
        SimulationResult with static risk metrics
    """
    loader = DataLoader()
    profile = build_profile_from_application(customer_application, loader)
    
    config = SimulationConfig(
        n_paths=n_paths,
        horizon_months=horizon_months,
        random_seed=random_seed,
        correlation_mode=CorrelationMode.CORRELATED
    )
    
    load = load_and_prepare(profile, config)
    
    result = run_simulation(profile, config, loan_config, load, scenario=None)
    
    return result


def compare_static_vs_dynamic(
    archetype_id: str,
    customer_application: CustomerApplication,
    loan_config: LoanConfig,
    random_seed: Optional[int] = None
) -> dict:
    """
    Compare static (Layer 1 only) vs dynamic (Layer 1 + Layer 2) simulations.
    
    Args:
        archetype_id: Archetype ID
        customer_application: Customer data
        loan_config: Loan parameters
        random_seed: Optional seed
    
    Returns:
        Dictionary with comparison metrics
    """
    static_result = run_static_simulation(
        customer_application,
        loan_config,
        random_seed=random_seed
    )
    
    dynamic_result = run_full_life_simulation(
        archetype_id,
        customer_application,
        loan_config,
        random_seed=random_seed
    )
    
    comparison = {
        "static": {
            "p_default": static_result.p_default,
            "expected_loss": static_result.expected_loss,
            "cvar_95": static_result.cvar_95,
            "risk_tier": static_result.recommended_loan.risk_tier.value,
            "approved": static_result.recommended_loan.approved
        },
        "dynamic": {
            "p_default": dynamic_result.p_default,
            "expected_loss": dynamic_result.expected_loss,
            "cvar_95": dynamic_result.cvar_95,
            "risk_tier": dynamic_result.recommended_loan.risk_tier.value,
            "approved": dynamic_result.recommended_loan.approved
        },
        "delta": {
            "p_default_change": dynamic_result.p_default - static_result.p_default,
            "expected_loss_change": dynamic_result.expected_loss - static_result.expected_loss,
            "risk_tier_changed": static_result.recommended_loan.risk_tier != dynamic_result.recommended_loan.risk_tier,
            "approval_changed": static_result.recommended_loan.approved != dynamic_result.recommended_loan.approved
        }
    }
    
    return comparison
