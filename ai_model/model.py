"""
VarLend AI Model - Main entry point for bank risk assessment.

Flow:
1. Take user prompt with default values (term length, loan amount, income, etc.)
2. Parse and validate user input
3. Create parameters and call life simulation + Monte Carlo models
4. Output JSON with simulation results
5. Generate 5-6 risk profile graphs for the bank
6. AI summarizes the output (1-2 paragraphs max)
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from life_simulation.run_life_simulation import run_full_life_simulation
from life_simulation.trajectory_builder import build_life_trajectory
from monte_carlo_sim.src.integration.profile_builder import CustomerApplication
from monte_carlo_sim.src.types import LoanConfig
from monte_carlo_sim.src.types import SimulationResult
from life_simulation.models import LifeTrajectory

try:
    from .llm_client import LLMClient
    from .config import Config
    from .validation import InputValidator
except ImportError:
    from llm_client import LLMClient
    from config import Config
    from validation import InputValidator

from ai_model.visualization.path_plotter import plot_income_paths, plot_income_distribution
from ai_model.visualization.risk_charts import plot_risk_summary_card, plot_default_timing_analysis
from ai_model.visualization.portfolio_charts import plot_portfolio_evolution, plot_income_evolution
from ai_model.visualization.event_timeline import plot_event_timeline
from ai_model.visualization.advanced_charts import (
    plot_risk_heatmap_matrix,
    plot_income_variance_funnel,
    plot_payment_burden_evolution
)
from data_pipeline.loaders import DataLoader


@dataclass
class BankRiskAssessment:
    """Complete risk assessment output for bank decision-making."""
    
    # Loan recommendation
    approved: bool
    risk_tier: str
    optimal_loan_amount: float
    optimal_loan_term: int
    optimal_loan_rate: float
    
    # Risk metrics
    default_probability: float
    expected_loss: float
    cvar_95: float
    
    # AI summary (1-2 paragraphs)
    executive_summary: str
    
    # Chart paths for bank dashboard
    charts: List[Dict[str, str]]
    
    # Full simulation data
    simulation_data: Dict[str, Any]
    
    # Metadata
    execution_time_seconds: float
    archetype_used: str
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)
    
    def save_to_file(self, filepath: Path):
        """Save assessment to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class VarLendModel:
    """
    Main VarLend model for risk assessment.
    
    Usage:
        model = VarLendModel()
        assessment = model.assess_loan_application(
            user_prompt="Gig worker with 3 platforms, wants $5000 loan",
            loan_amount=5000,
            loan_term_months=24,
            monthly_income=2500,
            platforms=["uber", "doordash"]
        )
        print(assessment.executive_summary)
        assessment.save_to_file("output.json")
    """
    
    def __init__(self):
        """Initialize the VarLend model."""
        self.validator = InputValidator()
        self.data_loader = DataLoader()
        self.llm_client = None
        
        # Try to initialize LLM client (optional - will use fallback if unavailable)
        try:
            self.llm_client = LLMClient()
            print(f"LLM initialized: {self.llm_client.get_provider_name()}")
        except Exception as e:
            print(f"LLM not available ({e}). Using rule-based summaries.")
        
        Config.ensure_output_dirs()
    
    def assess_loan_application(
        self,
        user_prompt: str = "",
        
        # Loan parameters (defaults)
        loan_amount: float = 5000,
        loan_term_months: int = 24,
        loan_rate: float = 0.12,
        
        # User financial data (defaults)
        monthly_income: float = 2000,
        platforms: List[str] = None,
        hours_per_week: float = 40,
        
        # Optional detailed data
        liquid_savings: Optional[float] = None,
        monthly_expenses: Optional[float] = None,
        existing_debt: Optional[float] = None,
        metro_area: str = "national",
        months_experience: int = 12,
        has_vehicle: bool = True,
        has_dependents: bool = False,
        
        # Simulation settings
        time_horizon_months: int = 60,
        n_paths: int = 5000,
        random_seed: int = 42,
        archetype_override: Optional[str] = None,
        
        # Output settings
        save_json: bool = True,
        json_output_path: Optional[Path] = None
        
    ) -> BankRiskAssessment:
        """
        Main entry point: Assess loan application and generate risk profile.
        
        Args:
            user_prompt: Natural language description of the applicant
            loan_amount: Requested loan amount ($)
            loan_term_months: Loan term in months
            loan_rate: Annual interest rate (e.g., 0.12 = 12%)
            monthly_income: Estimated monthly income ($)
            platforms: List of platforms user works on (e.g., ["uber", "doordash"])
            hours_per_week: Hours worked per week
            liquid_savings: Cash savings ($)
            monthly_expenses: Fixed monthly expenses ($)
            existing_debt: Monthly debt obligations ($)
            metro_area: Geographic area
            months_experience: Months as gig worker
            has_vehicle: Whether user owns a vehicle
            has_dependents: Whether user has dependents
            time_horizon_months: Simulation time horizon (months)
            n_paths: Number of Monte Carlo paths
            random_seed: Random seed for reproducibility
            archetype_override: Force specific archetype (optional)
            save_json: Whether to save full output to JSON
            json_output_path: Custom JSON output path
        
        Returns:
            BankRiskAssessment with risk metrics, AI summary, and chart paths
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print("VARLEND RISK ASSESSMENT")
        print("="*80)
        
        # STEP 1: Parse and validate input
        print("\n[1/6] Parsing and validating input...")
        
        if platforms is None:
            platforms = ["uber", "doordash"]
        
        user_data = self._parse_user_input(
            user_prompt=user_prompt,
            monthly_income=monthly_income,
            platforms=platforms,
            hours_per_week=hours_per_week,
            liquid_savings=liquid_savings,
            monthly_expenses=monthly_expenses,
            existing_debt=existing_debt,
            metro_area=metro_area,
            months_experience=months_experience,
            has_vehicle=has_vehicle,
            has_dependents=has_dependents
        )
        
        validation = self.validator.validate_user_data(user_data)
        if not validation.valid:
            raise ValueError(f"Invalid user data: {validation.missing_fields}")
        
        print(f"  [OK] User data validated")
        print(f"  [OK] Platforms: {', '.join(platforms)}")
        print(f"  [OK] Monthly income: ${monthly_income:,.0f}")
        print(f"  [OK] Loan request: ${loan_amount:,.0f} for {loan_term_months} months @ {loan_rate:.1%}")
        
        # STEP 2: Create parameters and select/build archetype
        print("\n[2/6] Creating simulation parameters...")
        
        archetype, archetype_name, is_custom = self._get_or_build_archetype(
            user_data, archetype_override
        )
        
        # Save custom archetype so trajectory builder can find it
        if is_custom:
            self._save_custom_archetype(archetype)
        
        customer_app = self._build_customer_application(
            archetype, user_data, loan_amount, loan_term_months
        )
        
        loan_config = LoanConfig(
            amount=loan_amount,
            term_months=loan_term_months,
            annual_rate=loan_rate
        )
        
        print(f"  [OK] Using archetype: {archetype_name}")
        print(f"  [OK] Base income mu: ${archetype['base_mu']:,.0f}")
        print(f"  [OK] Income volatility (CV): {archetype['coefficient_of_variation']:.1%}")
        
        # STEP 3: Run life simulation + Monte Carlo
        print(f"\n[3/6] Running simulation ({n_paths} paths, {time_horizon_months} months)...")
        print("  This may take 10-30 seconds...")
        
        # Build life trajectory for visualization
        trajectory = build_life_trajectory(
            archetype_id=archetype["id"],
            n_months=time_horizon_months,
            random_seed=random_seed
        )
        
        # Run full Monte Carlo simulation with life events
        result = run_full_life_simulation(
            archetype_id=archetype["id"],
            customer_application=customer_app,
            loan_config=loan_config,
            random_seed=random_seed,
            n_paths=n_paths,
            horizon_months=time_horizon_months
        )
        
        # #region agent log
        import json as log_json
        import numpy as np
        with open('debug-59f376.log', 'a') as log_f:
            log_f.write(log_json.dumps({
                "sessionId": "59f376",
                "location": "model.py:assess_loan_application:simulation_complete",
                "message": "Simulation results",
                "data": {
                    "p_default": float(result.p_default),
                    "expected_loss": float(result.expected_loss),
                    "median_income_first_6mo": [float(result.median_income_by_month[i]) for i in range(min(6, len(result.median_income_by_month)))],
                    "p10_income_first_6mo": [float(result.p10_income_by_month[i]) for i in range(min(6, len(result.p10_income_by_month)))],
                    "approved": result.recommended_loan.approved,
                    "risk_tier": result.recommended_loan.risk_tier.value
                },
                "timestamp": int(__import__('time').time() * 1000),
                "hypothesisId": "ALL"
            }) + '\n')
        # #endregion
        
        print(f"  [OK] Simulation completed")
        print(f"  [OK] Default probability: {result.p_default:.2%}")
        print(f"  [OK] Risk tier: {result.recommended_loan.risk_tier.value.upper()}")
        print(f"  [OK] Decision: {'APPROVED' if result.recommended_loan.approved else 'DECLINED'}")
        
        # STEP 4: Generate 9 risk profile graphs
        print("\n[4/6] Generating risk profile charts...")

        charts = self._generate_risk_charts(
            result=result,
            trajectory=trajectory,
            archetype=archetype,
            loan_config=loan_config,
            customer_app=customer_app
        )

        print(f"  [OK] Generated {len(charts)} charts:")
        for chart in charts:
            print(f"    - {chart['type']}: {chart['filename']}")
        
        # STEP 5: Create full simulation data JSON
        print("\n[5/6] Compiling simulation data...")
        
        simulation_data = self._compile_simulation_data(
            result=result,
            trajectory=trajectory,
            archetype=archetype,
            user_data=user_data,
            loan_config=loan_config
        )
        
        print(f"  [OK] Data compiled ({len(simulation_data)} top-level fields)")
        
        # STEP 6: Generate AI summary (1-2 paragraphs max)
        print("\n[6/6] Generating AI executive summary...")
        
        executive_summary = self._generate_executive_summary(
            result=result,
            trajectory=trajectory,
            archetype=archetype,
            loan_config=loan_config
        )
        
        word_count = len(executive_summary.split())
        print(f"  [OK] Summary generated ({word_count} words)")
        
        # Create final assessment
        execution_time = time.time() - start_time
        
        assessment = BankRiskAssessment(
            approved=result.recommended_loan.approved,
            risk_tier=result.recommended_loan.risk_tier.value,
            optimal_loan_amount=float(result.recommended_loan.optimal_amount),
            optimal_loan_term=int(result.recommended_loan.optimal_term_months),
            optimal_loan_rate=float(result.recommended_loan.optimal_rate),
            default_probability=float(result.p_default),
            expected_loss=float(result.expected_loss),
            cvar_95=float(result.cvar_95),
            executive_summary=executive_summary,
            charts=charts,
            simulation_data=simulation_data,
            execution_time_seconds=execution_time,
            archetype_used=archetype_name
        )
        
        # Save to JSON if requested
        if save_json:
            if json_output_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                json_output_path = Config.OUTPUT_DIR / f"assessment_{timestamp}.json"
            
            assessment.save_to_file(json_output_path)
            print(f"\n[OK] Assessment saved to: {json_output_path}")
        
        print("\n" + "="*80)
        print(f"ASSESSMENT COMPLETE - {execution_time:.1f}s")
        print("="*80)
        
        return assessment
    
    def _parse_user_input(
        self,
        user_prompt: str,
        monthly_income: float,
        platforms: List[str],
        hours_per_week: float,
        liquid_savings: Optional[float],
        monthly_expenses: Optional[float],
        existing_debt: Optional[float],
        metro_area: str,
        months_experience: int,
        has_vehicle: bool,
        has_dependents: bool
    ) -> Dict[str, Any]:
        """
        Parse user input into structured data.
        
        If LLM is available, use it to extract additional info from prompt.
        Otherwise, use provided parameters directly.
        """
        # Base user data from parameters
        user_data = {
            "platforms": [p.lower() for p in platforms],
            "hours_per_week": hours_per_week,
            "monthly_income_estimate": monthly_income,
            "metro_area": metro_area,
            "months_as_gig_worker": months_experience,
            "has_vehicle": has_vehicle,
            "has_dependents": has_dependents
        }
        
        # #region agent log
        import json as log_json
        with open('debug-59f376.log', 'a') as log_f:
            log_f.write(log_json.dumps({
                "sessionId": "59f376",
                "location": "model.py:parse_user_input:before_defaults",
                "message": "Input params before defaults",
                "data": {
                    "monthly_income": monthly_income,
                    "liquid_savings_provided": liquid_savings,
                    "monthly_expenses_provided": monthly_expenses,
                    "existing_debt_provided": existing_debt
                },
                "timestamp": int(__import__('time').time() * 1000),
                "hypothesisId": "A,B,E"
            }) + '\n')
        # #endregion
        
        # Add optional fields if provided
        if liquid_savings is not None:
            user_data["liquid_savings"] = liquid_savings
        else:
            # Default: 3 weeks of income
            user_data["liquid_savings"] = monthly_income * 0.75
        
        if monthly_expenses is not None:
            user_data["monthly_fixed_expenses"] = monthly_expenses
        else:
            # Default: 45% of income
            user_data["monthly_fixed_expenses"] = monthly_income * 0.45
        
        if existing_debt is not None:
            user_data["existing_debt_obligations"] = existing_debt
        else:
            # Default: 8% of income
            user_data["existing_debt_obligations"] = monthly_income * 0.08
        
        # #region agent log
        with open('debug-59f376.log', 'a') as log_f:
            log_f.write(log_json.dumps({
                "sessionId": "59f376",
                "location": "model.py:parse_user_input:after_defaults",
                "message": "Applied defaults",
                "data": {
                    "liquid_savings": user_data["liquid_savings"],
                    "monthly_fixed_expenses": user_data["monthly_fixed_expenses"],
                    "existing_debt_obligations": user_data["existing_debt_obligations"],
                    "total_monthly_obligations": user_data["monthly_fixed_expenses"] + user_data["existing_debt_obligations"]
                },
                "timestamp": int(__import__('time').time() * 1000),
                "hypothesisId": "A,B"
            }) + '\n')
        # #endregion
        
        # If user_prompt provided and LLM available, try to extract additional context
        if user_prompt and self.llm_client:
            try:
                enhanced_data = self._extract_from_prompt(user_prompt, user_data)
                user_data.update(enhanced_data)
            except Exception as e:
                print(f"  Note: Could not enhance from prompt ({e})")
        
        return user_data
    
    def _extract_from_prompt(self, prompt: str, base_data: Dict) -> Dict[str, Any]:
        """Use LLM to extract additional information from natural language prompt."""
        
        system_prompt = """You are parsing a loan application description for a gig worker.
        Extract any additional financial details mentioned in the prompt that aren't already provided.
        
        Return JSON with only the fields you find mentioned:
        {
            "monthly_income_estimate": float (if mentioned),
            "liquid_savings": float (if mentioned),
            "monthly_fixed_expenses": float (if mentioned),
            "existing_debt_obligations": float (if mentioned),
            "platforms": ["platform1", "platform2"] (if mentioned),
            "hours_per_week": float (if mentioned),
            "has_vehicle": bool (if mentioned),
            "has_dependents": bool (if mentioned),
            "metro_area": str (if mentioned)
        }
        
        Return empty {} if no additional info found. Return ONLY valid JSON.
        """
        
        user_prompt_text = f"""
        Current data: {json.dumps(base_data, indent=2)}
        
        Description: {prompt}
        
        Extract any additional details mentioned that would override or enhance the current data.
        """
        
        try:
            extracted = self.llm_client.generate_json(system_prompt, user_prompt_text, temperature=0.0)
            return extracted
        except:
            return {}
    
    def _get_or_build_archetype(
        self,
        user_data: Dict[str, Any],
        archetype_override: Optional[str]
    ) -> tuple[Dict[str, Any], str, bool]:
        """
        Get pre-defined archetype or build custom one from user data.
        
        Returns:
            (archetype_dict, archetype_name, is_custom)
        """
        # If override specified, use it
        if archetype_override:
            try:
                archetype = self.data_loader.load_archetype(archetype_override)
                return archetype, archetype["name"], False
            except:
                print(f"  Warning: Archetype '{archetype_override}' not found, auto-selecting...")
        
        # Build custom archetype from user data
        try:
            from .archetype_builder import ArchetypeBuilder
        except ImportError:
            from archetype_builder import ArchetypeBuilder
        
        builder = ArchetypeBuilder()
        archetype, validation = builder.build_custom_archetype(user_data)
        
        # #region agent log
        import json as log_json
        with open('debug-59f376.log', 'a') as log_f:
            log_f.write(log_json.dumps({
                "sessionId": "59f376",
                "location": "model.py:get_or_build_archetype:archetype_built",
                "message": "Custom archetype built",
                "data": {
                    "archetype_id": archetype["id"],
                    "archetype_name": archetype["name"],
                    "base_mu": archetype["base_mu"],
                    "base_sigma": archetype["base_sigma"],
                    "coefficient_of_variation": archetype["coefficient_of_variation"],
                    "platforms": archetype["platforms"]
                },
                "timestamp": int(__import__('time').time() * 1000),
                "hypothesisId": "C"
            }) + '\n')
        # #endregion
        
        return archetype, archetype["name"], True
    
    def _save_custom_archetype(self, archetype: Dict[str, Any]):
        """Save custom archetype to data pipeline."""
        import json
        
        archetypes_path = self.data_loader.data_dir / "archetypes.json"
        
        with open(archetypes_path, 'r') as f:
            data = json.load(f)
        
        # Check if archetype already exists
        existing_ids = [a["id"] for a in data["archetypes"]]
        if archetype["id"] not in existing_ids:
            data["archetypes"].append(archetype)
            data["metadata"]["count"] = len(data["archetypes"])
        else:
            # Update existing
            for i, a in enumerate(data["archetypes"]):
                if a["id"] == archetype["id"]:
                    data["archetypes"][i] = archetype
                    break
        
        with open(archetypes_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _build_customer_application(
        self,
        archetype: Dict[str, Any],
        user_data: Dict[str, Any],
        loan_amount: float,
        loan_term: int
    ) -> CustomerApplication:
        """Build CustomerApplication for Monte Carlo simulation."""
        
        platforms = archetype["platforms"]
        hours_per_week = archetype["hours_per_week"]
        experience = archetype["experience_months"]
        
        # Distribute hours evenly across platforms
        platforms_and_hours = [
            (platform, hours_per_week / len(platforms), experience)
            for platform in platforms
        ]
        
        # #region agent log
        import json as log_json
        annual_rate = 0.12
        monthly_rate = annual_rate / 12
        n_payments = loan_term
        monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
        
        with open('debug-59f376.log', 'a') as log_f:
            log_f.write(log_json.dumps({
                "sessionId": "59f376",
                "location": "model.py:build_customer_application:before_create",
                "message": "CustomerApplication parameters",
                "data": {
                    "platforms_and_hours": platforms_and_hours,
                    "liquid_savings": user_data["liquid_savings"],
                    "monthly_fixed_expenses": user_data["monthly_fixed_expenses"],
                    "existing_debt_obligations": user_data["existing_debt_obligations"],
                    "loan_request_amount": loan_amount,
                    "loan_term_months": loan_term,
                    "calculated_monthly_payment": float(monthly_payment),
                    "total_monthly_obligations": user_data["monthly_fixed_expenses"] + user_data["existing_debt_obligations"] + monthly_payment,
                    "archetype_base_mu": archetype["base_mu"]
                },
                "timestamp": int(__import__('time').time() * 1000),
                "hypothesisId": "B,D"
            }) + '\n')
        # #endregion
        
        return CustomerApplication(
            platforms_and_hours=platforms_and_hours,
            metro_area=archetype["metro"],
            months_as_gig_worker=experience,
            has_vehicle=user_data.get("has_vehicle", True),
            has_dependents=user_data.get("has_dependents", False),
            liquid_savings=user_data["liquid_savings"],
            monthly_fixed_expenses=user_data["monthly_fixed_expenses"],
            existing_debt_obligations=user_data["existing_debt_obligations"],
            loan_request_amount=loan_amount,
            requested_term_months=loan_term,
            acceptable_rate_range=(0.08, 0.20)
        )
    
    def _generate_risk_charts(
        self,
        result: SimulationResult,
        trajectory: LifeTrajectory,
        archetype: Dict[str, Any],
        loan_config: LoanConfig,
        customer_app = None
    ) -> List[Dict[str, str]]:
        """
        Generate 9 risk profile charts for bank dashboard.

        Charts generated:
        1. Income paths with percentile bands
        2. Risk summary card (key metrics)
        3. Default timing analysis
        4. Portfolio evolution (skill/platforms)
        5. Event timeline
        6. Income parameter evolution (mu/sigma/CV over time)
        7. Risk heatmap matrix (loan amount × term)
        8. Income variance funnel
        9. Payment burden evolution
        """
        charts = []
        
        try:
            # Chart 1: Income paths
            path = plot_income_paths(result, archetype)
            charts.append({
                "type": "income_paths",
                "filename": path.name,
                "full_path": str(path),
                "description": "Monte Carlo income trajectories with percentile bands"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate income_paths chart: {e}")
        
        try:
            # Chart 2: Risk summary card
            path = plot_risk_summary_card(result, archetype, loan_config)
            charts.append({
                "type": "risk_summary",
                "filename": path.name,
                "full_path": str(path),
                "description": "Risk assessment summary with key metrics"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate risk_summary chart: {e}")
        
        try:
            # Chart 3: Default timing analysis
            path = plot_default_timing_analysis(result, archetype)
            charts.append({
                "type": "default_timing",
                "filename": path.name,
                "full_path": str(path),
                "description": "When defaults occur across simulation paths"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate default_timing chart: {e}")
        
        try:
            # Chart 4: Portfolio evolution
            path = plot_portfolio_evolution(trajectory)
            charts.append({
                "type": "portfolio_evolution",
                "filename": path.name,
                "full_path": str(path),
                "description": "Skill growth and platform diversification over time"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate portfolio_evolution chart: {e}")
        
        try:
            # Chart 5: Event timeline
            path = plot_event_timeline(trajectory)
            charts.append({
                "type": "event_timeline",
                "filename": path.name,
                "full_path": str(path),
                "description": "Life events and their financial impacts"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate event_timeline chart: {e}")
        
        try:
            # Chart 6: Income evolution (mu, sigma, CV)
            path = plot_income_evolution(trajectory)
            charts.append({
                "type": "income_evolution",
                "filename": path.name,
                "full_path": str(path),
                "description": "Income parameter evolution (mean, volatility, CV)"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate income_evolution chart: {e}")
        
        # Advanced analytics charts
        try:
            # Chart 7: Risk heatmap matrix
            path = plot_risk_heatmap_matrix(result, customer_app, archetype["id"])
            charts.append({
                "type": "risk_matrix",
                "filename": path.name,
                "full_path": str(path),
                "description": "Default risk heatmap across loan structures"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate risk_matrix chart: {e}")
        
        try:
            # Chart 8: Income variance funnel
            path = plot_income_variance_funnel(result, archetype["id"])
            charts.append({
                "type": "variance_funnel",
                "filename": path.name,
                "full_path": str(path),
                "description": "Income uncertainty funnel showing distribution width"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate variance_funnel chart: {e}")
        
        try:
            # Chart 9: Payment burden evolution
            path = plot_payment_burden_evolution(result, customer_app, loan_config, archetype["id"])
            charts.append({
                "type": "payment_burden",
                "filename": path.name,
                "full_path": str(path),
                "description": "Loan payment as % of income over time"
            })
        except Exception as e:
            print(f"  Warning: Failed to generate payment_burden chart: {e}")

        return charts
    
    def _compile_simulation_data(
        self,
        result: SimulationResult,
        trajectory: LifeTrajectory,
        archetype: Dict[str, Any],
        user_data: Dict[str, Any],
        loan_config: LoanConfig
    ) -> Dict[str, Any]:
        """Compile full simulation data for JSON export."""
        
        return {
            "loan_recommendation": {
                "approved": result.recommended_loan.approved,
                "risk_tier": result.recommended_loan.risk_tier.value,
                "optimal_amount": float(result.recommended_loan.optimal_amount),
                "optimal_term_months": int(result.recommended_loan.optimal_term_months),
                "optimal_rate": float(result.recommended_loan.optimal_rate),
                "reasoning": result.recommended_loan.reasoning
            },
            "risk_metrics": {
                "default_probability": float(result.p_default),
                "expected_loss": float(result.expected_loss),
                "cvar_95": float(result.cvar_95),
                "median_income_by_month": result.median_income_by_month.tolist(),
                "p10_income_by_month": result.p10_income_by_month.tolist(),
                "p90_income_by_month": result.p90_income_by_month.tolist()
            },
            "trajectory_summary": {
                "months": trajectory.months,
                "n_events": len(trajectory.events),
                "macro_shock": {
                    "scenario": trajectory.macro_shock.scenario_name,
                    "start_month": trajectory.macro_shock.start_month
                } if trajectory.macro_shock else None,
                "initial_platforms": len(trajectory.portfolio_states[0].active_platforms),
                "final_platforms": len(trajectory.portfolio_states[-1].active_platforms),
                "initial_skill": float(trajectory.portfolio_states[0].skill_multiplier),
                "final_skill": float(trajectory.portfolio_states[-1].skill_multiplier),
                "narrative": trajectory.narrative
            },
            "events": [
                {
                    "month": e.month,
                    "type": e.event_type.value,
                    "description": e.description,
                    "income_impact": float(e.income_impact),
                    "expense_impact": float(e.expense_impact),
                    "duration_months": e.duration_months
                }
                for e in trajectory.events
            ],
            "archetype": {
                "id": archetype["id"],
                "name": archetype["name"],
                "base_income_mu": float(archetype["base_mu"]),
                "base_income_sigma": float(archetype["base_sigma"]),
                "coefficient_of_variation": float(archetype["coefficient_of_variation"]),
                "platforms": archetype["platforms"],
                "risk_category": archetype["default_risk_category"]
            },
            "user_profile": {
                "platforms": user_data["platforms"],
                "hours_per_week": user_data["hours_per_week"],
                "monthly_income": user_data["monthly_income_estimate"],
                "liquid_savings": user_data["liquid_savings"],
                "monthly_expenses": user_data["monthly_fixed_expenses"],
                "debt_obligations": user_data["existing_debt_obligations"],
                "metro_area": user_data["metro_area"]
            },
            "loan_request": {
                "amount": float(loan_config.amount),
                "term_months": int(loan_config.term_months),
                "annual_rate": float(loan_config.annual_rate)
            }
        }
    
    def _generate_executive_summary(
        self,
        result: SimulationResult,
        trajectory: LifeTrajectory,
        archetype: Dict[str, Any],
        loan_config: LoanConfig
    ) -> str:
        """
        Generate 1-2 paragraph executive summary using AI.
        
        If LLM unavailable, uses rule-based summary.
        """
        # Build context for LLM
        context = self._build_summary_context(result, trajectory, archetype, loan_config)
        
        if self.llm_client:
            try:
                return self._generate_llm_summary(context)
            except Exception as e:
                print(f"  Note: LLM summary failed ({e}), using rule-based summary")
        
        return self._generate_fallback_summary(context)
    
    def _build_summary_context(
        self,
        result: SimulationResult,
        trajectory: LifeTrajectory,
        archetype: Dict[str, Any],
        loan_config: LoanConfig
    ) -> Dict[str, Any]:
        """Build context dictionary for summary generation."""
        
        return {
            "approved": result.recommended_loan.approved,
            "p_default": result.p_default,
            "expected_loss": result.expected_loss,
            "cvar_95": result.cvar_95,
            "risk_tier": result.recommended_loan.risk_tier.value,
            "optimal_amount": result.recommended_loan.optimal_amount,
            "optimal_term": result.recommended_loan.optimal_term_months,
            "optimal_rate": result.recommended_loan.optimal_rate,
            "archetype_name": archetype["name"],
            "base_income": archetype["base_mu"],
            "income_cv": archetype["coefficient_of_variation"],
            "platforms": archetype["platforms"],
            "emergency_fund_weeks": archetype["emergency_fund_weeks"],
            "n_events": len(trajectory.events),
            "macro_shock": trajectory.macro_shock.scenario_name if trajectory.macro_shock else None,
            "time_horizon": trajectory.months,
            "reasoning": result.recommended_loan.reasoning
        }
    
    def _generate_llm_summary(self, context: Dict[str, Any]) -> str:
        """Generate AI summary using LLM (1-2 paragraphs max)."""
        
        system_prompt = """You are a financial analyst writing a brief executive summary for a bank's loan committee.

Your summary must be EXACTLY 1-2 paragraphs (max 150 words total).

Structure:
- Paragraph 1: Decision (approve/decline), default probability, and primary risk driver
- Paragraph 2: Key recommendation with optimal loan terms OR risk mitigation strategy

Be concise, professional, and actionable. Focus on the "why" behind the decision."""

        user_prompt = f"""
Loan Application Risk Assessment:

Decision: {'APPROVED' if context['approved'] else 'DECLINED'}
Default Probability: {context['p_default']:.2%}
Risk Tier: {context['risk_tier'].upper()}
Expected Loss: ${context['expected_loss']:,.0f}

Borrower Profile:
- Archetype: {context['archetype_name']}
- Monthly Income: ${context['base_income']:,.0f}
- Income Volatility (CV): {context['income_cv']:.1%}
- Platforms: {', '.join(context['platforms'])}
- Emergency Fund: {context['emergency_fund_weeks']} weeks
- Time Horizon: {context['time_horizon']} months

Trajectory:
- Life Events: {context['n_events']}
- Macro Shock: {context['macro_shock'] or 'None'}

Recommendation:
- Optimal Amount: ${context['optimal_amount']:,.0f}
- Optimal Term: {context['optimal_term']} months
- Optimal Rate: {context['optimal_rate']:.1%}
- Reasoning: {context['reasoning'][:2] if context['reasoning'] else []}

Write a 1-2 paragraph executive summary (max 150 words).
"""
        
        summary = self.llm_client.generate(system_prompt, user_prompt, temperature=0.2)
        
        # Enforce length limit
        words = summary.split()
        if len(words) > 150:
            summary = ' '.join(words[:150]) + "..."
        
        return summary.strip()
    
    def _generate_fallback_summary(self, context: Dict[str, Any]) -> str:
        """Generate rule-based summary when LLM unavailable."""
        
        lines = []
        
        # Paragraph 1: Decision and risk
        if context["approved"]:
            lines.append(
                f"This borrower is APPROVED with a {context['p_default']:.1%} default probability "
                f"over {context['time_horizon']} months ({context['risk_tier']} tier). "
            )
        else:
            lines.append(
                f"This borrower is DECLINED due to elevated default risk ({context['p_default']:.1%}) "
                f"over {context['time_horizon']} months. "
            )
        
        # Primary risk driver
        if context["income_cv"] > 0.35:
            lines.append(f"Primary concern is high income volatility (CV={context['income_cv']:.0%}). ")
        elif len(context["platforms"]) == 1:
            lines.append("Primary concern is single-platform concentration risk. ")
        elif context["emergency_fund_weeks"] < 3:
            lines.append(f"Primary concern is limited financial buffer ({context['emergency_fund_weeks']} weeks). ")
        else:
            lines.append("Risk profile is within acceptable parameters. ")
        
        # Paragraph 2: Recommendation
        lines.append("\n\n")
        
        if context["approved"]:
            lines.append(
                f"Recommended structure: ${context['optimal_amount']:,.0f} over "
                f"{context['optimal_term']} months at {context['optimal_rate']:.1%} APR. "
            )
            
            # Add key strength
            if len(context["platforms"]) >= 3:
                lines.append("Strong platform diversification reduces income correlation risk. ")
            elif context["emergency_fund_weeks"] >= 6:
                lines.append("Robust emergency fund provides cushion for income volatility. ")
            else:
                lines.append("Monitor for payment consistency during first 6 months. ")
        else:
            lines.append(
                f"Alternative approach: Consider smaller amount (${context['optimal_amount']:,.0f}) "
                f"with longer term ({context['optimal_term']} months) to reduce payment burden. "
            )
            
            if context["income_cv"] > 0.35:
                lines.append("Encourage platform diversification to reduce income volatility.")
            elif context["emergency_fund_weeks"] < 3:
                lines.append("Recommend building emergency fund before reapplying.")
        
        return ''.join(lines).strip()
    
    def print_summary(self, assessment: BankRiskAssessment):
        """Print formatted assessment summary to console."""
        
        print("\n" + "="*80)
        print("BANK RISK ASSESSMENT SUMMARY")
        print("="*80)
        
        print(f"\nDECISION: {'[APPROVED]' if assessment.approved else '[DECLINED]'}")
        print(f"Risk Tier: {assessment.risk_tier.upper()}")
        
        print("\nKEY METRICS:")
        print(f"  Default Probability: {assessment.default_probability:.2%}")
        print(f"  Expected Loss: ${assessment.expected_loss:,.2f}")
        print(f"  CVaR 95% (tail risk): ${assessment.cvar_95:,.2f}")
        
        print("\nRECOMMENDED LOAN STRUCTURE:")
        print(f"  Amount: ${assessment.optimal_loan_amount:,.0f}")
        print(f"  Term: {assessment.optimal_loan_term} months")
        print(f"  Rate: {assessment.optimal_loan_rate:.1%} APR")
        
        print("\n" + "-"*80)
        print("EXECUTIVE SUMMARY")
        print("-"*80)
        print(assessment.executive_summary)
        
        print("\n" + "-"*80)
        print(f"CHARTS GENERATED: {len(assessment.charts)}")
        print("-"*80)
        for chart in assessment.charts:
            print(f"  - {chart['type']}: {chart['filename']}")
        
        print("\n" + "="*80)


# Convenience function for simple usage
def assess_loan(
    loan_amount: float = 5000,
    loan_term_months: int = 24,
    monthly_income: float = 2000,
    platforms: List[str] = None,
    user_prompt: str = ""
) -> BankRiskAssessment:
    """
    Quick loan assessment with minimal parameters.
    
    Args:
        loan_amount: Requested loan amount ($)
        loan_term_months: Loan term in months
        monthly_income: Monthly income estimate ($)
        platforms: List of gig platforms (defaults to ["uber", "doordash"])
        user_prompt: Optional natural language description
    
    Returns:
        BankRiskAssessment
    
    Example:
        >>> assessment = assess_loan(
        ...     loan_amount=5000,
        ...     loan_term_months=24,
        ...     monthly_income=2500,
        ...     platforms=["uber", "doordash", "instacart"],
        ...     user_prompt="Experienced driver with 3 platforms, good savings"
        ... )
        >>> print(f"Decision: {'APPROVED' if assessment.approved else 'DECLINED'}")
        >>> print(f"Default risk: {assessment.default_probability:.1%}")
    """
    model = VarLendModel()
    
    assessment = model.assess_loan_application(
        user_prompt=user_prompt,
        loan_amount=loan_amount,
        loan_term_months=loan_term_months,
        monthly_income=monthly_income,
        platforms=platforms
    )
    
    model.print_summary(assessment)
    
    return assessment


if __name__ == "__main__":
    """
    Example usage: Run with default parameters.
    
    Run from command line:
        python -m ai_model.model
    """
    print("\nVarLend Model - Example Usage")
    print("="*80)
    
    # Example 1: Simple assessment with defaults
    print("\n\nEXAMPLE 1: Basic Loan Assessment")
    assessment = assess_loan(
        loan_amount=5000,
        loan_term_months=24,
        monthly_income=2500,
        platforms=["uber", "doordash"],
        user_prompt="Reliable driver with 2 platforms, 18 months experience"
    )
    
    # Example 2: Detailed assessment with all parameters
    print("\n\nEXAMPLE 2: Detailed Assessment")
    model = VarLendModel()
    assessment = model.assess_loan_application(
        user_prompt="Multi-platform driver in San Francisco, strong earnings",
        loan_amount=8000,
        loan_term_months=36,
        loan_rate=0.10,
        monthly_income=3500,
        platforms=["uber", "lyft", "doordash"],
        hours_per_week=45,
        liquid_savings=6000,
        monthly_expenses=2000,
        existing_debt=300,
        metro_area="san_francisco",
        months_experience=24,
        has_vehicle=True,
        has_dependents=False,
        time_horizon_months=60,
        n_paths=5000
    )
    
    model.print_summary(assessment)
    
    print("\n\n[OK] Demo complete! Check ai_model/outputs/ for JSON and charts.")
