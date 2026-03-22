"""
Simulation Runner - Orchestrates the full VarLend pipeline.

Coordinates archetype building, simulation execution, and result collection.
"""

import sys
from pathlib import Path
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from life_simulation.run_life_simulation import run_full_life_simulation
from life_simulation.trajectory_builder import build_life_trajectory
from data_pipeline.loaders import DataLoader
from monte_carlo_sim.src.integration.profile_builder import (
    CustomerApplication,
    build_profile_from_application,
    scenario_from_data_pipeline,
)
from monte_carlo_sim.src.engine.monte_carlo import _dominant_gig_type
from monte_carlo_sim.src.types import LoanConfig
from life_simulation.models import LifeTrajectory
from monte_carlo_sim.src.types import SimulationResult

from .archetype_builder import ArchetypeBuilder
from .parameter_extractor import SimulationRequest
from .validation import InputValidator
from .config import Config


@dataclass
class SimulationOutput:
    """Complete simulation output with trajectory and results."""
    
    run_id: str
    trajectory: LifeTrajectory
    result: SimulationResult
    archetype_used: dict
    is_custom_archetype: bool
    validation_warnings: list[str]
    execution_time_seconds: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory": {
                "archetype_id": self.trajectory.archetype_id,
                "months": self.trajectory.months,
                "n_events": len(self.trajectory.events),
                "events": [
                    {
                        "month": e.month,
                        "type": e.event_type.value,
                        "description": e.description,
                        "income_impact": float(e.income_impact),
                        "expense_impact": float(e.expense_impact)
                    }
                    for e in self.trajectory.events
                ],
                "portfolio_states": [
                    {
                        "month": p.month,
                        "platforms": p.active_platforms,
                        "skill_multiplier": float(p.skill_multiplier),
                        "monthly_income": float(p.monthly_base_income)
                    }
                    for p in self.trajectory.portfolio_states
                ],
                "macro_shock": {
                    "category": self.trajectory.macro_shock.category,
                    "scenario_name": self.trajectory.macro_shock.scenario_name,
                    "start_month": self.trajectory.macro_shock.start_month
                } if self.trajectory.macro_shock else None,
                "narrative": self.trajectory.narrative
            },
            "result": {
                "p_default": float(self.result.p_default),
                "expected_loss": float(self.result.expected_loss),
                "cvar_95": float(self.result.cvar_95),
                "median_income_by_month": self.result.median_income_by_month.tolist(),
                "p10_income_by_month": self.result.p10_income_by_month.tolist(),
                "p90_income_by_month": self.result.p90_income_by_month.tolist(),
                "time_to_default_percentiles": {
                    k: float(v) for k, v in self.result.time_to_default_percentiles.items()
                },
                "recommended_loan": {
                    "risk_tier": self.result.recommended_loan.risk_tier.value,
                    "optimal_amount": float(self.result.recommended_loan.optimal_amount),
                    "optimal_term_months": int(self.result.recommended_loan.optimal_term_months),
                    "optimal_rate": float(self.result.recommended_loan.optimal_rate),
                    "reasoning": self.result.recommended_loan.reasoning
                }
            },
            "archetype": {
                "id": self.archetype_used["id"],
                "name": self.archetype_used["name"],
                "is_custom": self.is_custom_archetype,
                "base_mu": float(self.archetype_used["base_mu"]),
                "base_sigma": float(self.archetype_used["base_sigma"]),
                "cv": float(self.archetype_used["coefficient_of_variation"]),
                "platforms": self.archetype_used["platforms"]
            },
            "metadata": {
                "validation_warnings": self.validation_warnings,
                "execution_time_seconds": self.execution_time_seconds
            }
        }


class SimulationRunner:
    """Orchestrates complete simulation pipeline."""
    
    def __init__(self):
        """Initialize simulation runner."""
        self.archetype_builder = ArchetypeBuilder()
        self.validator = InputValidator()
    
    def run_from_request(
        self,
        request: SimulationRequest,
        user_data: Optional[dict] = None,
        save_archetype: bool = False
    ) -> SimulationOutput:
        """
        Run simulation from structured request.
        
        Args:
            request: SimulationRequest from parameter extractor
            user_data: Optional user financial data
            save_archetype: Whether to save custom archetype to file
        
        Returns:
            SimulationOutput with complete results
        """
        import time
        start_time = time.time()
        run_id = f"{int(start_time * 1000)}"

        scenario = request.scenario
        
        archetype_base = scenario.get("archetype_base", "steady_sarah")
        custom_params = scenario.get("custom_params", {})
        
        if archetype_base == "custom" or user_data:
            if not user_data and custom_params:
                user_data = custom_params
            
            if not user_data:
                raise ValueError("Custom archetype requested but no user_data provided")
            
            archetype, is_custom = self.archetype_builder.load_or_build_archetype(
                archetype_id=None,
                user_data=user_data
            )
            validation_warnings = []
        else:
            archetype, is_custom = self.archetype_builder.load_or_build_archetype(
                archetype_id=archetype_base
            )
            validation_warnings = []
            
            if custom_params:
                archetype = self._apply_custom_params(archetype, custom_params)
                validation_warnings.append("Custom parameters applied to base archetype")
        
        if is_custom:
            self._save_custom_archetype(archetype)
        
        customer_app = self._build_customer_application(archetype, user_data, scenario)
        
        loan_config = LoanConfig(
            amount=scenario.get("loan_amount", 5000),
            term_months=scenario.get("loan_term_months", 24),
            annual_rate=scenario.get("loan_rate", 0.12)
        )
        
        time_horizon = scenario.get("time_horizon_months", Config.DEFAULT_TIME_HORIZON_MONTHS)
        n_paths = scenario.get("n_paths", Config.DEFAULT_N_PATHS)
        random_seed = scenario.get("random_seed", 42)

        trajectory = build_life_trajectory(
            archetype_id=archetype["id"],
            n_months=time_horizon,
            random_seed=random_seed,
        )

        merged_ai = self._build_merged_ai_scenario(scenario, customer_app, time_horizon)

        # Inject user's life shocks into the trajectory for display
        if merged_ai:
            trajectory = self._inject_user_events_into_trajectory(trajectory, merged_ai)
        print(f"[DEBUG] Raw scenario dict: {scenario}")
        print(f"[DEBUG] Merged AI scenario: {merged_ai}")
        if merged_ai is not None:
            print(f"[DEBUG] Scenario has {len(merged_ai.get('parameter_shifts', []))} shifts, {len(merged_ai.get('discrete_jumps', []))} jumps")
        if merged_ai is not None and not merged_ai.get("parameter_shifts") and not merged_ai.get("discrete_jumps"):
            print(f"[DEBUG] Scenario dropped - no parameter_shifts or discrete_jumps found")
            merged_ai = None

        result = run_full_life_simulation(
            archetype_id=archetype["id"],
            customer_application=customer_app,
            loan_config=loan_config,
            random_seed=random_seed,
            n_paths=n_paths,
            horizon_months=time_horizon,
            ai_scenario=merged_ai,
        )
        
        execution_time = time.time() - start_time
        
        return SimulationOutput(
            run_id=run_id,
            trajectory=trajectory,
            result=result,
            archetype_used=archetype,
            is_custom_archetype=is_custom,
            validation_warnings=validation_warnings,
            execution_time_seconds=execution_time
        )
    
    def run_comparison(
        self,
        request: SimulationRequest,
        user_data: Optional[dict] = None
    ) -> tuple[SimulationOutput, SimulationOutput]:
        """
        Run comparison between two scenarios.
        
        Args:
            request: SimulationRequest with scenario and scenario_b
            user_data: Optional user data
        
        Returns:
            (output_a, output_b) tuple
        """
        if not request.is_comparison():
            raise ValueError("Request is not a comparison request")
        
        original_scenario = request.scenario
        request.scenario = original_scenario
        output_a = self.run_from_request(request, user_data)
        
        request.scenario = request.scenario_b
        output_b = self.run_from_request(request, user_data)
        
        request.scenario = original_scenario
        
        return output_a, output_b
    
    def _apply_custom_params(self, archetype: dict, custom_params: dict) -> dict:
        """Apply custom parameter overrides to archetype."""
        archetype = archetype.copy()

        if "skill_growth_rate" in custom_params:
            archetype["skill_growth_rate"] = custom_params["skill_growth_rate"]

        if "platform_add_rate" in custom_params:
            archetype["platform_add_rate"] = custom_params["platform_add_rate"]

        if "platforms" in custom_params:
            archetype["platforms"] = custom_params["platforms"]

        if "hours_per_week" in custom_params:
            archetype["hours_per_week"] = custom_params["hours_per_week"]

        if "emergency_fund_weeks" in custom_params:
            archetype["emergency_fund_weeks"] = custom_params["emergency_fund_weeks"]

        return archetype

    def _inject_user_events_into_trajectory(
        self,
        trajectory: LifeTrajectory,
        ai_scenario: Dict[str, Any]
    ) -> LifeTrajectory:
        """
        Inject user-specified life shocks into the trajectory for display.

        This ensures the trajectory shown to users reflects their specified
        life events, not just randomly sampled ones.
        """
        from life_simulation.models import LifeEvent, EventType

        user_events = []

        # Convert discrete_jumps to LifeEvents
        for jump in ai_scenario.get("discrete_jumps", []):
            month = jump.get("month", 0)
            income_impact = jump.get("income_delta", 0)
            expense_impact = jump.get("expense_delta", 0)
            description = jump.get("label", "User-specified event")

            # Determine event type based on impacts
            if income_impact < -500:
                event_type = EventType.HEALTH_MAJOR_ILLNESS
            elif income_impact < 0:
                event_type = EventType.VEHICLE_MINOR_REPAIR
            elif expense_impact > 500:
                event_type = EventType.HOUSING_EMERGENCY_REPAIR
            elif expense_impact > 0:
                event_type = EventType.HOUSING_RENT_INCREASE
            else:
                event_type = EventType.POSITIVE_SIDE_GIG

            user_events.append(LifeEvent(
                event_type=event_type,
                month=month,
                income_impact=income_impact,
                expense_impact=-expense_impact,  # Expenses are negative in LifeEvent
                duration_months=1,
                description=description
            ))

        # Convert parameter_shifts to LifeEvents
        for shift in ai_scenario.get("parameter_shifts", []):
            month = shift.get("start_month", 0)
            duration = shift.get("duration_months", 1)
            mu_mult = shift.get("mu_multiplier", 1.0)
            sigma_mult = shift.get("sigma_multiplier", 1.0)
            expense_mult = shift.get("expense_multiplier", 1.0)

            # Determine event type and impact based on multipliers
            if mu_mult < 0.8:
                event_type = EventType.HEALTH_MAJOR_ILLNESS
                description = f"Major income disruption (Month {month}-{month+duration})"
                income_impact = -1000 * (1 - mu_mult)
            elif mu_mult < 1.0:
                event_type = EventType.PLATFORM_MARKET_SATURATION
                description = f"Reduced earnings (Month {month}-{month+duration})"
                income_impact = -500 * (1 - mu_mult)
            elif expense_mult > 1.2:
                event_type = EventType.HOUSING_RENT_INCREASE
                description = f"Increased expenses (Month {month}-{month+duration})"
                income_impact = 0
            else:
                event_type = EventType.POSITIVE_SKILL_UPGRADE
                description = f"Positive change (Month {month}-{month+duration})"
                income_impact = 500 * (mu_mult - 1)

            user_events.append(LifeEvent(
                event_type=event_type,
                month=month,
                income_impact=income_impact,
                expense_impact=0,
                duration_months=duration,
                description=description
            ))

        # Add user events to trajectory
        if user_events:
            # Sort all events by month
            all_events = trajectory.events + user_events
            all_events.sort(key=lambda e: e.month)
            trajectory.events = all_events

            # Update narrative to include user events
            user_narrative = ai_scenario.get("narrative", "")
            if user_narrative and trajectory.narrative:
                trajectory.narrative = f"{trajectory.narrative} | User scenario: {user_narrative}"
            elif user_narrative:
                trajectory.narrative = user_narrative

        return trajectory

    @staticmethod
    def _sanitize_ai_scenario_dict(raw: Dict[str, Any], horizon_months: int) -> Dict[str, Any]:
        shifts_out: List[Dict[str, Any]] = []
        for s in raw.get("parameter_shifts", []):
            if not isinstance(s, dict):
                continue
            try:
                start = int(s["start_month"])
                dur = int(s["duration_months"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or start >= horizon_months:
                continue
            dur = max(1, min(dur, horizon_months - start))
            shifts_out.append({**s, "start_month": start, "duration_months": dur})
        jumps_out: List[Dict[str, Any]] = []
        for j in raw.get("discrete_jumps", []):
            if not isinstance(j, dict):
                continue
            try:
                m = int(j["month"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= m < horizon_months:
                jumps_out.append(j)
        return {
            "narrative": str(raw.get("narrative", "")),
            "parameter_shifts": shifts_out,
            "discrete_jumps": jumps_out,
        }

    @staticmethod
    def _lookup_scenario_category(loader: DataLoader, scenario_name: str) -> Optional[str]:
        for cat, names in loader.list_scenarios().items():
            if scenario_name in names:
                return cat
        return None

    @staticmethod
    def _merge_scenario_dicts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        narratives: List[str] = []
        shifts: List[Dict[str, Any]] = []
        jumps: List[Dict[str, Any]] = []
        for p in parts:
            n = p.get("narrative")
            if n:
                narratives.append(str(n))
            shifts.extend(p.get("parameter_shifts", []) or [])
            jumps.extend(p.get("discrete_jumps", []) or [])
        return {
            "narrative": " | ".join(narratives) if narratives else "Scenario",
            "parameter_shifts": shifts,
            "discrete_jumps": jumps,
        }

    def _build_merged_ai_scenario(
        self,
        scenario: dict,
        customer_app: CustomerApplication,
        horizon_months: int,
    ) -> Optional[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        structured = scenario.get("structured_scenario")
        if structured and isinstance(structured, dict):
            parts.append(structured)
        forced = scenario.get("forced_events") or []
        if forced:
            loader = DataLoader()
            profile = build_profile_from_application(customer_app, loader)
            gig_type = _dominant_gig_type(profile)
            for fe in forced:
                if not isinstance(fe, dict):
                    continue
                ev_type = fe.get("type")
                if not ev_type:
                    continue
                start = int(fe.get("start_month", 0))
                cat = self._lookup_scenario_category(loader, str(ev_type))
                if not cat:
                    continue
                try:
                    raw = scenario_from_data_pipeline(
                        loader, cat, str(ev_type), start, gig_type
                    )
                    parts.append(raw)
                except Exception:
                    continue
        if not parts:
            return None
        merged = self._merge_scenario_dicts(parts)
        return self._sanitize_ai_scenario_dict(merged, horizon_months)
    
    def _build_customer_application(
        self,
        archetype: dict,
        user_data: Optional[dict],
        scenario: dict
    ) -> CustomerApplication:
        """Build CustomerApplication from archetype and user data."""
        
        platforms = archetype["platforms"]
        hours_per_week = archetype["hours_per_week"]
        experience = archetype["experience_months"]
        
        platforms_and_hours = [
            (platform, hours_per_week / len(platforms), experience)
            for platform in platforms
        ]
        
        base_mu = archetype["base_mu"]
        
        monthly_expenses = user_data.get("monthly_fixed_expenses") if user_data else None
        if monthly_expenses is None:
            monthly_expenses = base_mu * 0.45
        
        debt_obligations = user_data.get("existing_debt_obligations") if user_data else None
        if debt_obligations is None:
            debt_obligations = base_mu * 0.08
        
        liquid_savings = user_data.get("liquid_savings") if user_data else None
        if liquid_savings is None:
            liquid_savings = base_mu * 3.5
        
        loan_amount = scenario.get("loan_amount", archetype["recommended_loan_amount_range"][0])
        
        loan_term = scenario.get("loan_term_months", 48)
        
        metro = archetype["metro"]
        if user_data and user_data.get("metro_area"):
            metro = user_data["metro_area"]

        return CustomerApplication(
            platforms_and_hours=platforms_and_hours,
            metro_area=metro,
            months_as_gig_worker=experience,
            has_vehicle=user_data.get("has_vehicle", True) if user_data else True,
            has_dependents=user_data.get("has_dependents", False) if user_data else False,
            liquid_savings=liquid_savings,
            monthly_fixed_expenses=monthly_expenses,
            existing_debt_obligations=debt_obligations,
            loan_request_amount=loan_amount,
            requested_term_months=loan_term,
            acceptable_rate_range=(0.08, 0.20)
        )
    
    def _save_custom_archetype(self, archetype: dict):
        """Save custom archetype to data pipeline for trajectory building."""
        import json
        from data_pipeline.loaders import DataLoader
        
        loader = DataLoader()
        archetypes_path = loader.data_dir / "archetypes.json"
        
        with open(archetypes_path, 'r') as f:
            data = json.load(f)
        
        existing_ids = [a["id"] for a in data["archetypes"]]
        if archetype["id"] not in existing_ids:
            data["archetypes"].append(archetype)
            data["metadata"]["count"] = len(data["archetypes"])
        else:
            for i, a in enumerate(data["archetypes"]):
                if a["id"] == archetype["id"]:
                    data["archetypes"][i] = archetype
                    break
        
        with open(archetypes_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        output_dir = Config.OUTPUT_DIR / "custom_archetypes"
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / f"{archetype['id']}.json"
        with open(filepath, 'w') as f:
            json.dump(archetype, f, indent=2)


def run_simulation_from_query(
    query: str,
    user_data: Optional[dict] = None,
    loan_preferences: Optional[dict] = None
) -> SimulationOutput:
    """
    Convenience function to run simulation from natural language query.
    
    Args:
        query: Natural language query
        user_data: Optional user financial data
        loan_preferences: Optional loan preferences
    
    Returns:
        SimulationOutput
    
    Example:
        >>> output = run_simulation_from_query(
        ...     "Show me a 5 year path for a diversified worker",
        ...     user_data={"platforms": ["uber", "doordash"], "hours_per_week": 40}
        ... )
        >>> print(f"P(default): {output.result.p_default:.2%}")
    """
    from .parameter_extractor import ParameterExtractor
    
    extractor = ParameterExtractor()
    request = extractor.extract_with_context(query, user_data, loan_preferences)
    
    runner = SimulationRunner()
    return runner.run_from_request(request, user_data)
