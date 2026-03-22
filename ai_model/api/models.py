"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class UserData(BaseModel):
    """User financial data for custom archetype building."""
    
    platforms: List[str] = Field(..., description="List of gig platforms used")
    hours_per_week: float = Field(..., gt=0, le=80, description="Weekly hours worked")
    monthly_income_estimate: float = Field(..., gt=0, description="Estimated monthly income")
    
    metro_area: Optional[str] = Field(None, description="Metro area (national, san_francisco, etc.)")
    months_as_gig_worker: Optional[int] = Field(None, ge=0, description="Months of gig experience")
    has_vehicle: Optional[bool] = Field(True, description="Owns vehicle")
    has_dependents: Optional[bool] = Field(False, description="Has dependents")
    liquid_savings: Optional[float] = Field(None, ge=0, description="Cash savings")
    monthly_fixed_expenses: Optional[float] = Field(None, ge=0, description="Monthly fixed expenses")
    existing_debt_obligations: Optional[float] = Field(None, ge=0, description="Monthly debt payments")


class LoanPreferences(BaseModel):
    """Loan preference parameters."""
    
    amount: Optional[float] = Field(5000, gt=0, le=50000, description="Requested loan amount")
    term_months: Optional[int] = Field(24, ge=6, le=60, description="Loan term in months")
    max_rate: Optional[float] = Field(0.20, gt=0, le=0.50, description="Maximum acceptable rate")


class SimulateRequest(BaseModel):
    """Request for /api/simulate endpoint."""
    
    query: str = Field(..., min_length=10, max_length=2000, description="Natural language query")
    user_data: Optional[UserData] = Field(None, description="User financial data")
    loan_preferences: Optional[LoanPreferences] = Field(None, description="Loan preferences")
    
    use_archetype: Optional[str] = Field(None, description="Pre-defined archetype to use")
    random_seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    generate_charts: bool = Field(True, description="Generate visualization charts")
    structured_scenario: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional AIScenario from frontend/Claude (narrative, parameter_shifts, discrete_jumps)",
    )


class CompareRequest(BaseModel):
    """Request for /api/compare endpoint."""
    
    query: str = Field(..., min_length=10, description="Comparison query")
    user_data: Optional[UserData] = Field(None, description="Base user data")
    
    scenario_a_overrides: Optional[Dict[str, Any]] = Field(None, description="Overrides for scenario A")
    scenario_b_overrides: Optional[Dict[str, Any]] = Field(None, description="Overrides for scenario B")
    
    random_seed: Optional[int] = Field(42, description="Random seed")
    generate_charts: bool = Field(True, description="Generate charts")


class ValidateRequest(BaseModel):
    """Request for /api/validate endpoint."""
    
    user_data: UserData = Field(..., description="User data to validate")


class ChartInfo(BaseModel):
    """Information about a generated chart."""
    
    type: str = Field(..., description="Chart type (income_paths, risk_metrics, etc.)")
    path: str = Field(..., description="Relative path to chart file")
    description: str = Field(..., description="Chart description")


class SimulateResponse(BaseModel):
    """Response from /api/simulate endpoint."""
    
    summary: str = Field(..., description="AI-generated summary")
    quick_summary: str = Field(..., description="Quick 2-3 sentence summary")
    
    charts: List[ChartInfo] = Field(default_factory=list, description="Generated charts")
    
    metrics: Dict[str, Any] = Field(..., description="Key risk metrics")
    trajectory_info: Dict[str, Any] = Field(..., description="Trajectory summary")
    archetype_info: Dict[str, Any] = Field(..., description="Archetype used")
    
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    execution_time_seconds: float = Field(..., description="Execution time")


class CompareResponse(BaseModel):
    """Response from /api/compare endpoint."""
    
    comparison_summary: str = Field(..., description="Comparative analysis")
    
    scenario_a: Dict[str, Any] = Field(..., description="Scenario A results")
    scenario_b: Dict[str, Any] = Field(..., description="Scenario B results")
    
    charts: List[ChartInfo] = Field(default_factory=list, description="Comparison charts")
    
    delta_metrics: Dict[str, float] = Field(..., description="Difference in key metrics")
    winner: str = Field(..., description="Which scenario is preferable")


class ValidateResponse(BaseModel):
    """Response from /api/validate endpoint."""
    
    valid: bool = Field(..., description="Whether data is valid")
    missing_fields: List[str] = Field(default_factory=list, description="Required fields missing")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    defaults_applied: Dict[str, Any] = Field(default_factory=dict, description="Default values applied")
    
    closest_archetypes: Optional[List[Dict[str, Any]]] = Field(None, description="Closest matching archetypes")


class ArchetypeInfo(BaseModel):
    """Information about an archetype."""
    
    id: str
    name: str
    description: str
    base_mu: float
    coefficient_of_variation: float
    platforms: List[str]
    risk_category: str


class ArchetypesResponse(BaseModel):
    """Response from /api/archetypes endpoint."""
    
    archetypes: List[ArchetypeInfo] = Field(..., description="Available archetypes")
    count: int = Field(..., description="Number of archetypes")


class HealthResponse(BaseModel):
    """Response from /api/health endpoint."""
    
    status: str = Field(..., description="Service status")
    llm_provider: str = Field(..., description="Active LLM provider")
    version: str = Field("3.0", description="API version")
    components: Dict[str, bool] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions to fix")
