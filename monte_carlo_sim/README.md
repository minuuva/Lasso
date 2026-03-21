# VarLend Monte Carlo Simulation Engine

Credit risk engine for gig workers using forward-looking income simulation rather than backward-looking credit scores.

---

## What This Does

VarLend simulates **5,000 possible income futures** for a gig worker and determines whether they can repay a loan. Unlike traditional underwriting (which relies on credit history), this uses **stochastic modeling** to capture the inherent volatility of gig income.

**Core Innovation**: Distinguishes **variance** from **risk**. A worker earning $4,000/month ± $1,200 looks "unstable" to traditional lenders, but Monte Carlo can determine if that variance leads to actual defaults or if they have enough buffer to absorb it.

---

## Quick Start

### Installation

```bash
cd monte_carlo_sim
py -3 -m pip install -r requirements.txt
```

Requires: `numpy`, `scipy`, `python-dateutil`

### Run Demo

```bash
py -3 main.py
```

Output:
```
Baseline         | P(def)= 0.007 | EL=   20.36 | Tier=prime
Recession        | P(def)= 1.000 | EL= 2395.22 | Tier=decline
Injury           | P(def)= 0.093 | EL=  297.17 | Tier=subprime

Optimal: amount=8125.00, term=48m, rate=8.00%, P(def)=0.0054
```

### Run Integration Example

```bash
py -3 example_integration.py
```

Shows how to use `data_pipeline` parameters with customer-specific inputs.

---

## Project Structure

```
monte_carlo_sim/
├── src/
│   ├── types.py                      # All dataclasses and enums
│   ├── engine/                       # Core simulation
│   │   ├── monte_carlo.py            # Main orchestrator
│   │   ├── income_model.py           # Jump-diffusion draws
│   │   ├── parameter_state.py        # Time-varying parameter shifts
│   │   ├── correlation.py            # Multi-stream portfolio math
│   │   ├── seasonality.py            # Monthly income multipliers
│   │   └── defaults.py               # Vectorized default detection
│   ├── risk/
│   │   ├── risk_metrics.py           # P(default), CVaR, EL
│   │   └── loan_evaluator.py         # Tier assignment and recommendations
│   ├── ai/
│   │   └── scenario_parser.py        # Validates AI-produced scenarios
│   ├── data/
│   │   └── loader.py                 # JSON → WorkerProfile
│   └── integration/
│       └── profile_builder.py        # 🔗 Connects data_pipeline + customer data
├── data/
│   └── worker_profile.json           # Empty {} — teammate fills or use integration
├── main.py                           # Entry point (3 scenarios + sweep)
├── example_integration.py            # Full data pipeline integration example
├── INTEGRATION_GUIDE.md              # 📘 How to connect data sources
├── DATA_FLOW.md                      # 📊 Visual data architecture
├── PARAMETER_REFERENCE.md            # 📖 Parameter catalog
└── OUTPUT_EXPLAINED.md               # 📈 Understanding simulation results
```

---

## Three Ways to Use This

### Option 1: Standalone (Current `main.py`)

Use in-memory `WorkerProfile` objects for prototyping:

```python
from src.types import WorkerProfile, GigStream, GigType
from src.engine.monte_carlo import load_and_prepare, run_simulation

profile = WorkerProfile(
    streams=[GigStream(...)],
    liquid_savings=7500.0,
    # ... all fields
)
config = SimulationConfig(n_paths=5000, horizon_months=24)
load = load_and_prepare(profile, config)
result = run_simulation(profile, config, loan, load, scenario=None)

print(f"Decision: {result.recommended_loan.risk_tier.value}")
```

**Use when**: Quick testing, prototyping, demo without data pipeline.

### Option 2: JSON Loading (Via `loader.py`)

Load from `data/worker_profile.json`:

```python
from src.data.loader import load_worker_profile

load_result = load_worker_profile("data/worker_profile.json", config)
profile = load_result.profile
result = run_simulation(profile, config, loan, load_result, scenario=None)
```

**Use when**: Teammate provides pre-built JSON profiles.

### Option 3: Data Pipeline Integration (Via `profile_builder.py`)

Use research-backed parameters from `data_pipeline`:

```python
from data_pipeline.loaders import DataLoader
from src.integration.profile_builder import CustomerApplication, build_profile_from_application

# Customer submits application
application = CustomerApplication(
    platforms_and_hours=[("doordash", 35.0, 18), ("uber", 15.0, 12)],
    metro_area="atlanta",
    liquid_savings=6800.0,
    # ... all customer-specific fields
)

# Load research parameters
loader = DataLoader()

# Build WorkerProfile (combines customer + research)
profile = build_profile_from_application(application, loader)

# Run simulation
result = run_simulation(profile, config, loan, load, scenario=None)
```

**Use when**: Production system integrating real customer applications with vetted research data.

---

## Key Features

### 1. Fully Vectorized (No Python Path Loops)

All 5,000 paths are updated simultaneously using NumPy array operations:

```python
# Month-by-month updates for ALL paths at once
income[:, t] = rng.normal(mu_t, sigma_t, n_paths)  # Shape: (5000,)
buffer = np.maximum(buffer + net_cash_flow, 0.0)   # Shape: (5000,)
defaulted |= (rolling_stress & low_buffer)         # Shape: (5000,)
```

**Performance**: 5,000 paths × 24 months = 120,000 draws in ~10 seconds.

### 2. Jump-Diffusion Income Model

Goes beyond simple Gaussian:

```python
Income = N(μ, σ)                    # ← Normal variation
       + Poisson_Jump(λ=0.25)       # ← Discrete shocks (car repair, illness)
       + Echo                       # ← Shock carry-forward (repair takes 2 weeks)
       + Discrete_Event             # ← AI scenario shocks
       |> max(0)                    # ← Can't earn negative income
```

**Why**: Normal distribution underestimates **tail risk**. Real gig income has fat tails (rare but severe negative shocks).

### 3. Multi-Stream Correlation

Workers with Uber + DoorDash aren't independent — recession hits both:

```python
# Correlation matrix
       DD    Uber
DD  [ 1.0   0.4  ]
Uber[ 0.4   1.0  ]

# Portfolio sigma (lower than naive sum)
σ_portfolio = sqrt(w^T Σ w) < w · σ
```

**Result**: Diversification benefit is **quantified** — multi-platform workers get credit for lower risk.

### 4. Time-Varying Parameters

Income isn't constant — it changes with:

- **Seasonality**: December delivery income is 1.35× July
- **Macro shocks**: Recession drops income 25% over 18 months
- **Discrete events**: Injury in month 4 causes 2-month income drop

All modeled month-by-month with deterministic or stochastic rules.

### 5. Realistic Default Logic

Default requires **both**:
1. Persistent shortfall (rolling 3-month net cash < −1.5 × payment)
2. Depleted reserves (buffer < 1.0 × payment)

**Why**: One bad month doesn't cause default if you have savings. But **sustained** bleeding + **no buffer** does.

---

## Documentation

| Document | Purpose |
|----------|---------|
| **INTEGRATION_GUIDE.md** | How to connect data pipeline with Monte Carlo engine |
| **DATA_FLOW.md** | Visual diagrams of parameter flow |
| **PARAMETER_REFERENCE.md** | Complete catalog of all parameters (fixed vs. changeable) |
| **OUTPUT_EXPLAINED.md** | How to interpret P(default), CVaR, tiers, etc. |

**Start here**: `INTEGRATION_GUIDE.md` → then `OUTPUT_EXPLAINED.md`

---

## Example: Complete Loan Decision Flow

```python
# 1. Customer applies
application = CustomerApplication(
    platforms_and_hours=[("doordash", 30, 18)],
    metro_area="atlanta",
    liquid_savings=7000,
    monthly_fixed_expenses=1200,
    loan_request_amount=5000,
    requested_term_months=24,
    # ... other fields
)

# 2. Load research parameters
from data_pipeline.loaders import DataLoader
loader = DataLoader()

# 3. Build WorkerProfile
from src.integration.profile_builder import build_profile_from_application
profile = build_profile_from_application(application, loader)

# 4. Run Monte Carlo
from src.engine.monte_carlo import load_and_prepare, run_simulation
from src.types import SimulationConfig, LoanConfig

config = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=42)
load = load_and_prepare(profile, config)
loan = LoanConfig(amount=5000, term_months=24, annual_rate=0.14)

result = run_simulation(profile, config, loan, load, scenario=None)

# 5. Make decision
rec = result.recommended_loan
if rec.approved:
    print(f"✓ APPROVED: {rec.risk_tier.value.upper()}")
    print(f"  Amount: ${rec.optimal_amount:,.0f}")
    print(f"  Rate: {rec.optimal_rate:.1%}")
    print(f"  P(default): {result.p_default:.2%}")
else:
    print(f"✗ DECLINED: P(default) = {result.p_default:.1%} (too high)")
    if rec.alternative_structures:
        alt = rec.alternative_structures[0]
        print(f"  Alternative: ${alt['amount']:,.0f} @ {alt['p_default']:.1%} default risk")
```

---

## Testing

### Unit Tests (Not Yet Implemented)

```bash
pytest tests/
```

Would test:
- Correlation matrix construction
- Jump-diffusion draws
- Default detection logic
- Seasonality lookups

### Integration Test

```bash
py -3 example_integration.py
```

Tests:
- Data pipeline → ProfileBuilder → Monte Carlo → Risk classification
- Macro scenario application
- Loan sweep optimization

### Validation

Compare Monte Carlo output to **historical loan performance data** (when available):
- Predicted P(default) vs. actual default rate
- Predicted time-to-default distribution vs. actual
- CVaR vs. worst-case observed losses

Recalibrate jump parameters (λ, μ_jump, σ_jump) to match empirical default patterns.

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single simulation (5k paths × 24 months) | ~10 sec | Vectorized NumPy |
| Loan sweep (100 configs) | ~15 min | 100 simulations serially |
| Scenario stress test (3 scenarios) | ~30 sec | 3 simulations |

**Optimization**: For production, run sweep in parallel (distribute 100 configs across workers).

---

## Architecture Philosophy

### Three-Layer Design

1. **Data Pipeline** (`data_pipeline/`)
   - Research-backed universal parameters
   - Updated quarterly when new studies publish
   - No customer data

2. **Integration Layer** (`src/integration/profile_builder.py`)
   - Pure function: `(customer_data, pipeline_data) → WorkerProfile`
   - Deterministic, testable, auditable
   - No business logic

3. **Monte Carlo Engine** (`src/engine/`, `src/risk/`)
   - Stochastic simulation and risk metrics
   - Doesn't know about "platforms" or "metros" — just (μ, σ) inputs
   - Stable core

**Result**: Clean separation of concerns. Research team maintains pipeline. Product team collects customer data. Engine team maintains simulation math. Integration layer connects them.

---

## FAQ

**Q: How do I add a new customer?**  
A: Use `CustomerApplication` with their data + `build_profile_from_application`. See `example_integration.py`.

**Q: How do I change platform hourly rates?**  
A: Edit `data_pipeline/ingest/static_params.py` → `PLATFORM_EARNINGS`. This updates for ALL customers.

**Q: How do I test a different loan amount for the same customer?**  
A: Change `LoanConfig(amount=...)` and re-run `run_simulation`. No need to rebuild profile.

**Q: What if I don't have the data pipeline?**  
A: Use Option 1 (standalone) — manually create `WorkerProfile` objects. See `main.py` lines 34-67.

**Q: Can I use historical income data for a specific customer?**  
A: Yes. Fit (μ, σ) from their income history, optionally override jump parameters in JSON.

**Q: How do I stress-test for a future recession?**  
A: Use `scenario_from_data_pipeline(loader, "recession", "recession_2008", ...)` to apply historical recession shocks.

**Q: Why is my customer getting declined?**  
A: Check if `income - expenses - loan_payment < 0`. If their net cash flow is negative, they **mathematically cannot** repay any loan.

---

## Credits

**Research Sources**:
- JPMorgan Chase Institute (income volatility)
- Gridwise (platform earnings)
- FRED (macroeconomic data)
- Triplog (expense structure)

**Built for**: VarLend Hackathon Project

---

## Next Steps

1. **Understand the output**: Read `OUTPUT_EXPLAINED.md`
2. **Learn the data flow**: Read `INTEGRATION_GUIDE.md`
3. **Explore parameters**: Read `PARAMETER_REFERENCE.md`
4. **Run your first simulation**: Modify `example_integration.py` with your customer data
5. **Integrate with your app**: Use `profile_builder.build_profile_from_application` in your API endpoint
