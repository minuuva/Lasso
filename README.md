# Lasso - Variable Income Lending Risk Assessment

AI-powered loan risk assessment system for gig economy workers with variable income streams.

## Quick Start for UI Integration

### Run the Complete Test

```bash
py test_complete_pipeline.py
```

This generates:
- **13 charts total**: 9 standard + 4 advanced 3D visualizations
- Risk assessment JSON with approval decision, risk tier, default probability
- Executive summary from AI

### Use in Your UI

```python
from ai_model.model import VarLendModel

model = VarLendModel()

assessment = model.assess_loan_application(
    user_prompt="Your natural language prompt here...",
    loan_amount=5000,
    loan_term_months=24,
    monthly_income=3600,  # GROSS income
    platforms=["uber", "doordash"],
    hours_per_week=40,
    liquid_savings=1800,
    monthly_expenses=350,  # Should be realistic relative to NET income
    existing_debt=100,
    # Optional params...
)

# Returns BankRiskAssessment with:
# - assessment.approved (bool)
# - assessment.risk_tier (str)
# - assessment.default_probability (float)
# - assessment.executive_summary (str)
# - assessment.charts (list of chart metadata)
# - assessment.simulation_data (dict with full metrics)
```

### Output Structure

Charts are saved to: `ai_model/outputs/charts/`

**Standard Charts (9):**
1. `income_paths_*.png` - Monte Carlo income trajectories
2. `risk_summary_*.png` - Key risk metrics card
3. `default_timing_*.png` - When defaults occur
4. `portfolio_evolution_*.png` - Asset/debt evolution
5. `event_timeline_*.png` - Life events impact
6. `income_evolution_*.png` - Income parameters over time
7. `risk_matrix_*.png` - Risk heatmap
8. `variance_funnel_*.png` - Income uncertainty funnel
9. `payment_burden_*.png` - Payment to income ratio

**Advanced 3D Charts (4):**
10. `risk_surface_3d_*.png` - 3D risk surface
11. `volatility_surface_3d_*.png` - 3D volatility surface
12. `stress_test_matrix_*.png` - Stress test scenarios
13. `default_waterfall_*.png` - Default probability waterfall

## Architecture

- **ai_model/** - Main AI layer and orchestration
- **monte_carlo_sim/** - Core income simulation engine
- **life_simulation/** - Life trajectory and event modeling
- **data_pipeline/** - Data loading and archetype management

## Key Components

- `ai_model/model.py` - Main VarLendModel class
- `ai_model/api/server.py` - FastAPI REST API
- `ai_model/visualization/*` - All chart generation
- `test_complete_pipeline.py` - Reference implementation for UI

## Requirements

See root `requirements.txt` (covers the full project, including `monte_carlo_sim`).

## Important Notes

- Users report **GROSS income**
- System calculates **NET income** internally for simulation
- Expenses and debt should be realistic relative to NET income (~25-35% of net)
- The simulation uses platform-specific hourly rates to calculate actual net income
