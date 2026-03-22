"""
FastAPI Server - REST API for VarLend AI Layer.

Provides endpoints for natural language simulation queries.
"""

import sys
import os
from pathlib import Path
from typing import List
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    SimulateRequest, SimulateResponse,
    CompareRequest, CompareResponse,
    ValidateRequest, ValidateResponse,
    ArchetypesResponse, ArchetypeInfo,
    HealthResponse, ErrorResponse,
    ChartInfo
)
from .middleware import setup_cors, setup_rate_limiting

from ai_model.parameter_extractor import ParameterExtractor
from ai_model.simulation_runner import SimulationRunner
from ai_model.result_summarizer import ResultSummarizer
from ai_model.archetype_builder import ArchetypeBuilder
from ai_model.validation import InputValidator
from ai_model.config import Config
from ai_model.llm_client import LLMClient

from data_pipeline.loaders import DataLoader


app = FastAPI(
    title="VarLend AI Layer API",
    description="Natural language interface for gig worker risk assessment",
    version="3.0.0"
)

setup_cors(app)
setup_rate_limiting(app, max_requests=20, window_seconds=60)

if Config.CHART_DIR.exists():
    app.mount("/charts", StaticFiles(directory=str(Config.CHART_DIR)), name="charts")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "VarLend AI Layer API",
        "version": "3.0.0",
        "status": "operational",
        "endpoints": {
            "simulate": "POST /api/simulate - Run simulation from natural language query",
            "compare": "POST /api/compare - Compare two scenarios",
            "validate": "POST /api/validate - Validate user data",
            "archetypes": "GET /api/archetypes - List available archetypes",
            "health": "GET /api/health - Check service health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API and component health."""
    
    try:
        llm_client = LLMClient()
        llm_provider = llm_client.get_provider_name()
        llm_available = True
    except:
        llm_provider = "none"
        llm_available = False
    
    components = {
        "llm": llm_available,
        "data_pipeline": True,
        "monte_carlo": True,
        "life_simulation": True,
        "visualization": True
    }
    
    overall_status = "healthy" if all(components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        llm_provider=llm_provider,
        version="3.0",
        components=components
    )


@app.get("/api/archetypes", response_model=ArchetypesResponse)
async def get_archetypes():
    """Get list of available pre-defined archetypes."""
    
    try:
        loader = DataLoader()
        archetype_ids = ["volatile_vic", "steady_sarah", "weekend_warrior", "sf_hustler", "rising_ryan"]
        
        archetypes = []
        for arch_id in archetype_ids:
            try:
                arch = loader.load_archetype(arch_id)
                archetypes.append(ArchetypeInfo(
                    id=arch["id"],
                    name=arch["name"],
                    description=arch["description"],
                    base_mu=arch["base_mu"],
                    coefficient_of_variation=arch["coefficient_of_variation"],
                    platforms=arch["platforms"],
                    risk_category=arch["default_risk_category"]
                ))
            except:
                continue
        
        return ArchetypesResponse(
            archetypes=archetypes,
            count=len(archetypes)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load archetypes: {str(e)}"
        )


@app.post("/api/validate", response_model=ValidateResponse)
async def validate_user_data(request: ValidateRequest):
    """Validate user financial data."""
    
    try:
        validator = InputValidator()
        user_data_dict = request.user_data.dict(exclude_none=True)
        
        validation = validator.validate_user_data(user_data_dict)
        
        closest_archetypes = None
        if validation.valid:
            try:
                builder = ArchetypeBuilder()
                similarities = builder.compare_to_archetypes(user_data_dict)
                
                loader = DataLoader()
                closest_archetypes = []
                for arch_id, similarity in similarities[:3]:
                    arch = loader.load_archetype(arch_id)
                    closest_archetypes.append({
                        "id": arch_id,
                        "name": arch["name"],
                        "similarity_score": float(similarity),
                        "description": arch["description"]
                    })
            except:
                pass
        
        return ValidateResponse(
            valid=validation.valid,
            missing_fields=validation.missing_fields,
            warnings=validation.warnings,
            suggestions=validation.suggestions,
            defaults_applied=validation.defaults_applied,
            closest_archetypes=closest_archetypes
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {str(e)}"
        )


@app.post("/api/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest):
    """
    Run simulation from natural language query.
    
    Main endpoint for AI-powered risk assessment.
    """
    
    try:
        start_time = time.time()
        
        validator = InputValidator()
        query_validation = validator.validate_query(request.query)
        
        if not query_validation.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid query: {query_validation.missing_fields}"
            )
        
        user_data_dict = None
        if request.user_data:
            user_data_dict = request.user_data.dict(exclude_none=True)
        
        loan_prefs_dict = None
        if request.loan_preferences:
            loan_prefs_dict = request.loan_preferences.dict(exclude_none=True)
        
        extractor = ParameterExtractor()
        sim_request = extractor.extract_with_context(
            request.query,
            user_data_dict,
            loan_prefs_dict
        )
        
        if request.use_archetype:
            sim_request.scenario["archetype_base"] = request.use_archetype
        
        if request.random_seed:
            sim_request.scenario["random_seed"] = request.random_seed

        if request.structured_scenario:
            sim_request.scenario["structured_scenario"] = request.structured_scenario
        
        runner = SimulationRunner()
        output = runner.run_from_request(sim_request, user_data_dict)
        
        summarizer = ResultSummarizer()
        summary = summarizer.summarize(output)
        quick_summary = summarizer.generate_quick_summary(output)
        
        charts = []
        if request.generate_charts:
            charts = _generate_all_charts(output)
        
        execution_time = time.time() - start_time
        
        return SimulateResponse(
            summary=summary,
            quick_summary=quick_summary,
            charts=charts,
            metrics={
                "p_default": float(output.result.p_default),
                "expected_loss": float(output.result.expected_loss),
                "cvar_95": float(output.result.cvar_95),
                "risk_tier": output.result.recommended_loan.risk_tier.value,
                "approved": output.result.recommended_loan.approved
            },
            trajectory_info={
                "months": output.trajectory.months,
                "n_events": len(output.trajectory.events),
                "macro_shock": output.trajectory.macro_shock.scenario_name if output.trajectory.macro_shock else None,
                "narrative": output.trajectory.narrative
            },
            archetype_info={
                "id": output.archetype_used["id"],
                "name": output.archetype_used["name"],
                "is_custom": output.is_custom_archetype,
                "cv": float(output.archetype_used["coefficient_of_variation"]),
                "platforms": output.archetype_used["platforms"]
            },
            warnings=output.validation_warnings,
            execution_time_seconds=execution_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@app.post("/api/compare", response_model=CompareResponse)
async def compare_scenarios(request: CompareRequest):
    """
    Compare two simulation scenarios.
    
    Useful for "what-if" analysis and scenario comparison.
    """
    
    try:
        extractor = ParameterExtractor()
        sim_request = extractor.extract_parameters(request.query)
        
        if not sim_request.is_comparison():
            sim_request.mode = "compare"
            
            if not sim_request.scenario_b:
                sim_request.scenario_b = sim_request.scenario.copy()
                
                if request.scenario_b_overrides:
                    sim_request.scenario_b.update(request.scenario_b_overrides)
        
        if request.scenario_a_overrides:
            sim_request.scenario.update(request.scenario_a_overrides)
        
        user_data_dict = None
        if request.user_data:
            user_data_dict = request.user_data.dict(exclude_none=True)
        
        runner = SimulationRunner()
        output_a, output_b = runner.run_comparison(sim_request, user_data_dict)
        
        summarizer = ResultSummarizer()
        comparison_summary = summarizer.summarize_comparison(
            output_a, output_b,
            scenario_a_description="Scenario A",
            scenario_b_description="Scenario B"
        )
        
        charts = []
        if request.generate_charts:
            from ai_model.visualization.comparison_plots import plot_comparison
            comp_chart = plot_comparison(output_a, output_b)
            charts.append(ChartInfo(
                type="comparison",
                path=f"/charts/{comp_chart.name}",
                description="Side-by-side scenario comparison"
            ))
        
        delta_default = output_b.result.p_default - output_a.result.p_default
        delta_loss = output_b.result.expected_loss - output_a.result.expected_loss
        
        winner = "scenario_a" if output_a.result.p_default < output_b.result.p_default else "scenario_b"
        if abs(delta_default) < 0.02:
            winner = "similar"
        
        return CompareResponse(
            comparison_summary=comparison_summary,
            scenario_a=output_a.to_dict(),
            scenario_b=output_b.to_dict(),
            charts=charts,
            delta_metrics={
                "delta_p_default": float(delta_default),
                "delta_expected_loss": float(delta_loss),
                "delta_cvar": float(output_b.result.cvar_95 - output_a.result.cvar_95)
            },
            winner=winner
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


def _generate_all_charts(output) -> List[ChartInfo]:
    """Generate all visualization charts for simulation output."""
    
    from ai_model.visualization.path_plotter import (
        plot_income_paths,
        plot_income_distribution,
        plot_net_cash_flow,
    )
    from ai_model.visualization.risk_charts import (
        plot_default_timing_analysis, plot_risk_summary_card
    )
    from ai_model.visualization.portfolio_charts import (
        plot_portfolio_evolution,
        plot_income_evolution,
    )
    from ai_model.visualization.event_timeline import (
        plot_event_timeline,
        plot_event_impact_summary,
    )
    from monte_carlo_sim.src.types import LoanConfig
    
    charts = []
    
    arch = output.archetype_used
    result = output.result
    trajectory = output.trajectory
    
    loan_config = LoanConfig(
        amount=result.recommended_loan.optimal_amount,
        term_months=result.recommended_loan.optimal_term_months,
        annual_rate=result.recommended_loan.optimal_rate
    )
    
    try:
        path = plot_income_paths(result, arch)
        charts.append(ChartInfo(
            type="income_paths",
            path=f"/charts/{path.name}",
            description="Monte Carlo income trajectories with percentile bands"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate income_paths chart: {e}")
    
    try:
        path = plot_risk_summary_card(result, arch, loan_config)
        charts.append(ChartInfo(
            type="risk_summary",
            path=f"/charts/{path.name}",
            description="Risk assessment summary card"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate risk_summary chart: {e}")
    
    try:
        path = plot_portfolio_evolution(trajectory)
        charts.append(ChartInfo(
            type="portfolio_evolution",
            path=f"/charts/{path.name}",
            description="Platform and skill growth over time"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate portfolio_evolution chart: {e}")
    
    try:
        path = plot_event_timeline(trajectory)
        charts.append(ChartInfo(
            type="event_timeline",
            path=f"/charts/{path.name}",
            description="Life event timeline with impacts"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate event_timeline chart: {e}")
    
    try:
        path = plot_default_timing_analysis(result, arch)
        charts.append(ChartInfo(
            type="default_timing",
            path=f"/charts/{path.name}",
            description="When defaults occur analysis"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate default_timing chart: {e}")

    try:
        path = plot_income_distribution(result, arch)
        charts.append(ChartInfo(
            type="income_distribution",
            path=f"/charts/{path.name}",
            description="Income distribution at month 0"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate income_distribution chart: {e}")

    try:
        path = plot_net_cash_flow(result, arch, loan_config)
        charts.append(ChartInfo(
            type="net_cash_flow",
            path=f"/charts/{path.name}",
            description="Net cash flow after expenses and loan payment"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate net_cash_flow chart: {e}")

    try:
        path = plot_income_evolution(trajectory)
        charts.append(ChartInfo(
            type="income_evolution",
            path=f"/charts/{path.name}",
            description="Income mean and volatility evolution over life trajectory"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate income_evolution chart: {e}")

    try:
        path = plot_event_impact_summary(trajectory)
        charts.append(ChartInfo(
            type="event_impact_summary",
            path=f"/charts/{path.name}",
            description="Summary of life event impacts by category"
        ))
    except Exception as e:
        print(f"Warning: Failed to generate event_impact_summary chart: {e}")
    
    return charts


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "suggestions": ["Check API logs for details", "Verify input data format"]
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting VarLend AI Layer API...")
    print(f"Server: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"Docs: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    
    uvicorn.run(
        "server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
