# GigSim: Monte Carlo Life Simulator for Gig Worker Credit Risk

## HooHacks 2026 | Finance Track | Capital One Judges

---

## 1. WHAT WE'RE BUILDING

A Monte Carlo simulation engine that models the financial life of a gig economy worker over 1-5 years. The simulation generates thousands of possible income paths accounting for platform earnings volatility, seasonality, macro shocks, and life events. The output is used to evaluate whether a gig worker can service a specific loan, and to compare our risk assessment against what a traditional FICO-based model would predict.

The core thesis: traditional credit models (FICO, DTI) systematically misprice gig worker risk because they assume stable income. Our simulation captures the actual income distribution, which lets us identify gig workers who are more reliable than FICO suggests AND ones who are more fragile than FICO suggests.

### Target User Personas
1. **Loan officer at Capital One** - wants to know: "Should I approve this gig worker for an auto loan?"
2. **The gig worker themselves** - wants to know: "Can I actually afford this loan, given my income swings?"
3. **Risk analyst** - wants to stress test: "What happens to this borrower if gas prices spike 40%?"

---

## 2. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                   │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Profile  │  │  Fan Chart   │  │  Loan Evaluation  │  │
│  │  Input    │  │  (Recharts)  │  │  Panel            │  │
│  │  Form     │  │              │  │                   │  │
│  └────┬─────┘  └──────▲───────┘  └────────▲──────────┘  │
│       │               │                   │              │
│       │         ┌─────┴───────────────────┘              │
│       ▼         │                                        │
│  ┌──────────────┴─────────────────────────────────────┐  │
│  │              API Route / Server Action              │  │
│  └──────────────┬─────────────────────────────────────┘  │
└─────────────────┼────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (Python FastAPI)                │
│                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │  Monte Carlo  │  │  Loan         │  │  AI Scenario  │  │
│  │  Engine       │  │  Evaluator    │  │  Agent        │  │
│  │  (NumPy)      │  │               │  │  (Gemini/     │  │
│  │               │  │               │  │   Claude)     │  │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│  ┌──────▼──────────────────▼──────────────────▼───────┐  │
│  │              Data Layer (JSON files + FRED API)     │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts (fan charts + distributions), shadcn/ui components
- **Backend**: Python 3.11+, FastAPI, NumPy (simulation), httpx (API calls)
- **AI Agent**: Google Gemini API (free tier) or Claude API for natural language scenario injection
- **Data**: Pre-cached JSON files + live FRED API calls
- **Deployment**: Vercel (frontend) + Railway or Modal (backend) OR run both locally for demo

---

## 3. MONTE CARLO ENGINE (core - build this first)

### 3.1 Gig Worker Profile (Input Schema)

```python
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class Platform(str, Enum):
    UBER_RIDESHARE = "uber_rideshare"
    UBER_EATS = "uber_eats"
    DOORDASH = "doordash"
    INSTACART = "instacart"
    LYFT = "lyft"
    TASKRABBIT = "taskrabbit"
    MULTI_PLATFORM = "multi_platform"

class MetroArea(str, Enum):
    # Pre-cache data for these metros for the demo
    NEW_YORK = "new_york"
    LOS_ANGELES = "los_angeles"
    CHICAGO = "chicago"
    SAN_FRANCISCO = "san_francisco"
    WASHINGTON_DC = "washington_dc"
    RICHMOND = "richmond"  # Cap1 HQ city - smart inclusion

class GigWorkerProfile(BaseModel):
    platform: Platform
    metro: MetroArea
    hours_per_week: float  # typical hours (will vary stochastically)
    months_experience: int  # affects earnings stability
    has_secondary_income: bool = False
    secondary_monthly_income: float = 0  # W-2 or other stable income
    monthly_rent: float
    monthly_fixed_expenses: float  # insurance, phone, subscriptions
    current_savings: float
    credit_score: int  # for FICO comparison, not used in our model
    dependents: int = 0
```

### 3.2 Simulation Parameters

```python
class SimulationConfig(BaseModel):
    num_paths: int = 5000          # number of Monte Carlo paths
    horizon_months: int = 36       # simulation length
    seed: Optional[int] = None     # for reproducibility
```

### 3.3 Monthly Income Model

The core stochastic process for monthly gig income:

```
Monthly_Gross_Income = Base_Rate(platform, metro) 
                       × Hours_Worked(month)
                       × Seasonal_Multiplier(month)
                       × Macro_Adjustment(unemployment, inflation)
                       × Platform_Demand_Noise
                       + Tips(base_rate, platform)

Monthly_Net_Income   = Monthly_Gross_Income
                       - Gas_Cost(miles_driven × gas_price)
                       - Vehicle_Maintenance(stochastic)
                       - Self_Employment_Tax(15.3% of net earnings)
                       - Platform_Fees(already netted in base_rate for simplicity)

Monthly_Cash_Flow    = Monthly_Net_Income
                       + Secondary_Income
                       - Rent
                       - Fixed_Expenses
                       - Life_Shock(if triggered)
                       - Loan_Payment(if evaluating)
```

### 3.4 Calibration Data (hardcoded from research)

These numbers come from JPMorgan Chase Institute research and BLS data. Cite them in the demo.

```python
# --- INCOME VOLATILITY PARAMETERS ---
# Source: JPMC Institute "Weathering Volatility 2.0" (2019)
MEDIAN_MONTHLY_CV = 0.36          # coefficient of variation for month-to-month income
INCOME_SPIKE_PROBABILITY = 0.33   # ~4 months/year have spikes (spikes 2x as likely as dips)
INCOME_DIP_PROBABILITY = 0.17     # ~2 months/year have dips
SPIKE_MAGNITUDE_MEAN = 0.30       # average spike is +30% of mean
DIP_MAGNITUDE_MEAN = 0.25         # average dip is -25% of mean

# --- SEASONAL MULTIPLIERS (by month, index 0 = January) ---
# Source: JPMC Institute finding that spikes cluster in March and December
# Plus known gig economy patterns (holiday surge, Jan slump)
SEASONAL_MULTIPLIERS = [
    0.85,  # Jan - post-holiday slump
    0.90,  # Feb
    1.10,  # Mar - tax refund spending boost, spring break
    1.00,  # Apr
    1.00,  # May
    1.05,  # Jun - summer travel
    1.05,  # Jul
    0.95,  # Aug
    0.95,  # Sep
    1.05,  # Oct - pre-holiday ramp
    1.10,  # Nov - holiday surge begins
    1.20,  # Dec - peak holiday demand
]

# --- PLATFORM BASE RATES (hourly, after platform fees, before expenses) ---
# Source: Gridwise 2025, Berkeley Labor Center, EPI
PLATFORM_BASE_RATES = {
    "uber_rideshare": {"mean": 23.33, "std": 5.0},
    "uber_eats":      {"mean": 18.00, "std": 6.0},
    "doordash":       {"mean": 17.50, "std": 5.5},
    "instacart":      {"mean": 19.00, "std": 5.0},
    "lyft":           {"mean": 21.00, "std": 5.0},
    "taskrabbit":     {"mean": 28.00, "std": 10.0},
    "multi_platform": {"mean": 21.00, "std": 6.0},
}

# --- METRO COST-OF-LIVING MULTIPLIERS ---
# Affects both earnings and expenses
METRO_MULTIPLIERS = {
    "new_york":       {"earnings": 1.25, "gas_price": 3.80, "rent_index": 1.50},
    "los_angeles":    {"earnings": 1.15, "gas_price": 4.50, "rent_index": 1.35},
    "chicago":        {"earnings": 1.05, "gas_price": 3.60, "rent_index": 1.10},
    "san_francisco":  {"earnings": 1.30, "gas_price": 4.80, "rent_index": 1.60},
    "washington_dc":  {"earnings": 1.15, "gas_price": 3.50, "rent_index": 1.25},
    "richmond":       {"earnings": 0.95, "gas_price": 3.30, "rent_index": 0.85},
}

# --- EXPENSE PARAMETERS ---
GAS_GALLONS_PER_HOUR = 1.2           # average for rideshare (city driving, ~25mpg, 30mi/hr)
MAINTENANCE_MONTHLY_BASE = 75.0       # average monthly maintenance
MAINTENANCE_SHOCK_PROB = 0.08         # 8% chance per month of a major repair
MAINTENANCE_SHOCK_RANGE = (500, 2500) # major repair costs
SELF_EMPLOYMENT_TAX_RATE = 0.153
TIP_PERCENTAGE = {                    # tips as % of gross, varies by platform
    "uber_rideshare": 0.15,
    "uber_eats": 0.25,
    "doordash": 0.30,
    "instacart": 0.20,
    "lyft": 0.15,
    "taskrabbit": 0.10,
    "multi_platform": 0.20,
}

# --- LIFE SHOCK EVENTS ---
# Poisson-distributed random events
LIFE_SHOCKS = [
    {"name": "medical_bill", "monthly_prob": 0.03, "cost_range": (500, 5000)},
    {"name": "car_accident", "monthly_prob": 0.01, "cost_range": (1000, 8000), "income_loss_months": 1},
    {"name": "illness", "monthly_prob": 0.04, "cost_range": (200, 1000), "income_loss_pct": 0.5, "duration_months": 1},
    {"name": "family_emergency", "monthly_prob": 0.02, "cost_range": (500, 3000)},
    {"name": "phone_breaks", "monthly_prob": 0.03, "cost_range": (200, 1200)},  # gig workers NEED their phone
    {"name": "deactivation_scare", "monthly_prob": 0.02, "income_loss_pct": 1.0, "duration_months": 1, "cost_range": (0, 0)},
]

# --- HOURS WORKED DISTRIBUTION ---
# Hours per week vary stochastically around the stated target
HOURS_CV = 0.20  # coefficient of variation for weekly hours
MIN_HOURS_WEEK = 0
MAX_HOURS_WEEK = 70
```

### 3.5 Simulation Engine (pseudocode)

```python
import numpy as np

def run_simulation(profile: GigWorkerProfile, config: SimulationConfig, macro_scenario: dict = None) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    
    n = config.num_paths
    T = config.horizon_months
    
    # Pre-compute base parameters
    platform = PLATFORM_BASE_RATES[profile.platform]
    metro = METRO_MULTIPLIERS[profile.metro]
    base_hourly = platform["mean"] * metro["earnings"]
    hourly_std = platform["std"] * metro["earnings"]
    
    # Initialize output arrays
    monthly_net_income = np.zeros((n, T))
    monthly_cash_flow = np.zeros((n, T))
    savings_balance = np.zeros((n, T))
    savings_balance[:, 0] = profile.current_savings
    
    for t in range(T):
        month_of_year = t % 12
        seasonal = SEASONAL_MULTIPLIERS[month_of_year]
        
        # --- MACRO ADJUSTMENT ---
        macro_mult = 1.0
        if macro_scenario and t in macro_scenario:
            macro_mult = macro_scenario[t].get("demand_multiplier", 1.0)
        
        # --- HOURS WORKED (per month, ~4.33 weeks) ---
        weekly_hours = rng.normal(profile.hours_per_week, profile.hours_per_week * HOURS_CV, size=n)
        weekly_hours = np.clip(weekly_hours, MIN_HOURS_WEEK, MAX_HOURS_WEEK)
        monthly_hours = weekly_hours * 4.33
        
        # --- GROSS INCOME ---
        hourly_rate = rng.normal(base_hourly, hourly_std, size=n)
        hourly_rate = np.maximum(hourly_rate, 8.0)  # floor at rough minimum
        
        gross = hourly_rate * monthly_hours * seasonal * macro_mult
        
        # --- INCOME VOLATILITY SHOCKS (JPMC-calibrated) ---
        # Random spikes and dips on top of base volatility
        spike_mask = rng.random(n) < INCOME_SPIKE_PROBABILITY / 12  # monthly
        dip_mask = rng.random(n) < INCOME_DIP_PROBABILITY / 12
        spike_amount = rng.normal(SPIKE_MAGNITUDE_MEAN, 0.10, size=n)
        dip_amount = rng.normal(DIP_MAGNITUDE_MEAN, 0.10, size=n)
        gross[spike_mask] *= (1 + np.abs(spike_amount[spike_mask]))
        gross[dip_mask] *= (1 - np.abs(dip_amount[dip_mask]))
        
        # --- TIPS ---
        tip_rate = TIP_PERCENTAGE.get(profile.platform, 0.15)
        tips = gross * rng.normal(tip_rate, tip_rate * 0.3, size=n)
        tips = np.maximum(tips, 0)
        
        # --- EXPENSES ---
        gas_price = macro_scenario[t].get("gas_price", metro["gas_price"]) if (macro_scenario and t in macro_scenario) else metro["gas_price"]
        gas_cost = monthly_hours * GAS_GALLONS_PER_HOUR * gas_price
        
        maintenance = np.full(n, MAINTENANCE_MONTHLY_BASE)
        shock_mask = rng.random(n) < MAINTENANCE_SHOCK_PROB
        maintenance[shock_mask] += rng.uniform(*MAINTENANCE_SHOCK_RANGE, size=shock_mask.sum())
        
        se_tax = np.maximum(gross + tips - gas_cost - maintenance, 0) * SELF_EMPLOYMENT_TAX_RATE
        
        # --- NET INCOME ---
        net = gross + tips - gas_cost - maintenance - se_tax
        monthly_net_income[:, t] = net
        
        # --- LIFE SHOCKS ---
        life_shock_cost = np.zeros(n)
        for shock in LIFE_SHOCKS:
            triggered = rng.random(n) < shock["monthly_prob"]
            if triggered.any():
                costs = rng.uniform(*shock["cost_range"], size=triggered.sum())
                life_shock_cost[triggered] += costs
                if "income_loss_pct" in shock:
                    net[triggered] *= (1 - shock["income_loss_pct"])
        
        # --- CASH FLOW ---
        total_expenses = profile.monthly_rent + profile.monthly_fixed_expenses + life_shock_cost
        cash_flow = net + profile.secondary_monthly_income - total_expenses
        monthly_cash_flow[:, t] = cash_flow
        
        # --- SAVINGS ---
        if t > 0:
            savings_balance[:, t] = np.maximum(savings_balance[:, t-1] + cash_flow, 0)
        else:
            savings_balance[:, t] = np.maximum(profile.current_savings + cash_flow, 0)
    
    return SimulationResult(
        monthly_net_income=monthly_net_income,
        monthly_cash_flow=monthly_cash_flow,
        savings_balance=savings_balance,
        percentiles=compute_percentiles(monthly_net_income),
    )

def compute_percentiles(data: np.ndarray) -> dict:
    """Compute percentile bands for fan chart visualization."""
    return {
        "p10": np.percentile(data, 10, axis=0).tolist(),
        "p25": np.percentile(data, 25, axis=0).tolist(),
        "p50": np.percentile(data, 50, axis=0).tolist(),
        "p75": np.percentile(data, 75, axis=0).tolist(),
        "p90": np.percentile(data, 90, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
    }
```

### 3.6 Simulation Output Schema (what the API returns to frontend)

```python
class SimulationResult(BaseModel):
    # Percentile bands for fan chart (each is array of length horizon_months)
    income_percentiles: dict  # {p10, p25, p50, p75, p90, mean}
    cash_flow_percentiles: dict
    savings_percentiles: dict
    
    # Loan evaluation (if loan params provided)
    loan_evaluation: Optional[LoanEvaluation]
    
    # Summary statistics
    summary: SimulationSummary

class SimulationSummary(BaseModel):
    mean_monthly_income: float
    median_monthly_income: float
    income_volatility_cv: float  # actual CV from simulation
    prob_negative_cash_flow_any_month: float  # % of paths with at least 1 negative month
    prob_savings_depleted: float  # % of paths where savings hit $0
    median_savings_at_end: float
    worst_month_p10: float  # 10th percentile of worst single month across paths
    
class LoanEvaluation(BaseModel):
    # Loan inputs
    loan_amount: float
    annual_rate: float
    term_months: int
    monthly_payment: float
    
    # Our model's assessment
    prob_miss_one_payment: float      # P(miss >= 1 payment in loan term)
    prob_miss_three_consecutive: float # P(90+ day delinquency)
    prob_default: float               # P(miss >= 6 payments or savings depleted + negative CF for 3+ months)
    months_to_first_miss_p50: float   # median months until first missed payment (among paths that miss)
    
    # FICO comparison
    fico_estimated_default_rate: float  # lookup from standard FICO-to-default tables
    fico_score: int
    risk_delta: float  # our_default_rate - fico_default_rate (negative = we think they're safer)
    risk_assessment: str  # "FICO_OVERESTIMATES_RISK" | "FICO_UNDERESTIMATES_RISK" | "ALIGNED"
```

---

## 4. LOAN EVALUATION MODULE

### 4.1 Loan Input

```python
class LoanParams(BaseModel):
    amount: float          # e.g. 15000
    annual_rate: float     # e.g. 0.072 for 7.2%
    term_months: int       # e.g. 60
```

### 4.2 Logic

For each Monte Carlo path:
1. Compute monthly loan payment using standard amortization formula
2. Subtract payment from monthly cash flow
3. Track savings balance (buffer absorbs negative months)
4. Flag "missed payment" when cash_flow < 0 AND savings_balance <= 0
5. Flag "default" when missed payments >= 6 OR (savings depleted AND negative cash flow for 3+ consecutive months)

### 4.3 FICO Comparison Table

Use the standard FICO-to-default-rate mapping (approximate, from public Fed data):

```python
FICO_DEFAULT_RATES = {
    # FICO range: annual default probability
    (300, 579): 0.28,
    (580, 619): 0.18,
    (620, 659): 0.11,
    (660, 699): 0.06,
    (700, 739): 0.03,
    (740, 799): 0.01,
    (800, 850): 0.005,
}
```

---

## 5. AI SCENARIO AGENT

### 5.1 Purpose

Translates natural language stress scenarios into structured parameter overrides for the Monte Carlo engine. The simulation stays mathematically rigorous; the AI just makes it accessible.

### 5.2 Implementation

Single LLM call. Input: user's natural language scenario. Output: structured JSON parameter overrides.

```python
SCENARIO_SYSTEM_PROMPT = """
You are a financial scenario translator. The user will describe an economic scenario in natural language. 
Your job is to translate it into structured parameter adjustments for a Monte Carlo simulation of gig worker income.

Output ONLY valid JSON with this schema:
{
  "scenario_name": "string",
  "description": "one sentence summary",
  "duration_months": int,
  "start_month": int (0-indexed from simulation start),
  "adjustments": {
    "demand_multiplier": float (1.0 = normal, 0.7 = 30% demand drop),
    "gas_price_override": float or null (dollars per gallon),
    "unemployment_rate": float or null,
    "hours_reduction_pct": float or null (0.0 to 1.0),
    "extra_expense_monthly": float or null,
    "tip_multiplier": float or null (1.0 = normal)
  }
}

Examples:
- "recession" -> demand_multiplier: 0.75, tip_multiplier: 0.8, hours_reduction_pct: 0.15
- "gas prices hit $6" -> gas_price_override: 6.0
- "Uber cuts driver pay 15%" -> demand_multiplier: 0.85
- "I get injured for 2 months" -> hours_reduction_pct: 1.0, duration_months: 2, extra_expense_monthly: 3000
"""
```

### 5.3 Integration Flow

1. User types scenario in text box
2. Frontend sends to `/api/scenario` endpoint
3. Backend calls Gemini/Claude with the system prompt + user input
4. Parse JSON response into `MacroScenario` object
5. Re-run simulation with scenario applied
6. Return delta results (before vs. after scenario)

---

## 6. DATA LAYER

### 6.1 Pre-cached Data Files (store in /data directory)

Download these BEFORE the hackathon and store as JSON:

```
/data
  /fred
    gas_prices.json          # GASREGW series, weekly, last 5 years
    unemployment.json        # UNRATE series, monthly
    cpi.json                 # CPIAUCSL series, monthly  
    fed_funds_rate.json      # FEDFUNDS series, monthly
    consumer_delinquency.json # DRCCLACBS series, quarterly
  /calibration
    platform_earnings.json   # compiled from Gridwise, EPI, Berkeley research
    metro_adjustments.json   # cost of living multipliers by metro
    jpmc_volatility.json     # JPMC Institute volatility parameters
    fico_default_table.json  # FICO score to default rate mapping
    seasonal_patterns.json   # 12-month seasonal multiplier curves by platform
  /scenarios
    recession_2008.json      # historical parameter replay
    recession_2020.json      # COVID shock parameters
    gas_spike_2022.json      # 2022 gas price surge
```

### 6.2 Live API Calls (optional, for demo polish)

FRED API (free key from fred.stlouisfed.org):
- Current gas price: `GASREGW`
- Current unemployment: `UNRATE`
- Current CPI: `CPIAUCSL`

Use these to show "current economic conditions" in the dashboard. Fall back to cached data if API is down.

---

## 7. API ENDPOINTS

### 7.1 FastAPI Routes

```python
# POST /api/simulate
# Runs the core Monte Carlo simulation
# Input: GigWorkerProfile + SimulationConfig + optional LoanParams
# Output: SimulationResult

# POST /api/simulate/with-loan  
# Same as above but requires LoanParams, returns LoanEvaluation
# Input: GigWorkerProfile + SimulationConfig + LoanParams
# Output: SimulationResult with LoanEvaluation populated

# POST /api/scenario
# Translates natural language to scenario parameters via AI agent
# Input: { "prompt": "What if gas hits $6/gallon for 3 months?" }
# Output: MacroScenario JSON

# POST /api/simulate/compare
# Runs simulation twice: baseline vs. with scenario applied
# Input: GigWorkerProfile + SimulationConfig + LoanParams + MacroScenario
# Output: { baseline: SimulationResult, stressed: SimulationResult, delta: DeltaSummary }

# GET /api/data/current-conditions
# Returns current macro data from FRED (or cache fallback)
# Output: { gas_price, unemployment_rate, cpi, fed_funds_rate, as_of_date }
```

---

## 8. FRONTEND SPEC (for the frontend developer)

### 8.1 Pages

**Single Page App** with these sections/panels:

1. **Profile Input Panel** (left sidebar or top section)
   - Platform dropdown (Uber, DoorDash, etc.)
   - Metro area dropdown
   - Hours per week slider (5-70)
   - Monthly rent input
   - Monthly fixed expenses input
   - Current savings input
   - Credit score input (for FICO comparison)
   - Has secondary income toggle + amount
   - Dependents counter

2. **Simulation Controls**
   - "Run Simulation" button
   - Time horizon selector (12, 24, 36, 48, 60 months)
   - Number of paths (1000, 5000, 10000)

3. **Fan Chart Panel** (main visualization, center)
   - X-axis: months
   - Y-axis: monthly net income (dollars)
   - Bands: p10-p90 (lightest), p25-p75 (medium), p50 median line (dark)
   - Should look like a spreading fan / cone of uncertainty
   - Color: use a blue gradient (Cap1 brand adjacent)

4. **Loan Evaluation Panel** (right side or below fan chart)
   - Loan amount input
   - Interest rate input
   - Term dropdown (24, 36, 48, 60 months)
   - "Evaluate Loan" button
   - Output cards:
     - Monthly payment amount
     - Our model: default probability (big number, color coded green/yellow/red)
     - FICO model: estimated default rate
     - Delta: "Our model says this borrower is X% safer/riskier than FICO predicts"
     - Risk assessment badge: "FICO OVERESTIMATES RISK" (green) / "ALIGNED" (yellow) / "FICO UNDERESTIMATES RISK" (red)

5. **Scenario Injection Panel** (bottom or expandable drawer)
   - Text input: "Describe a scenario to stress test..."
   - Preset buttons: "2008 Recession", "Gas Spike", "Platform Pay Cut", "Injury"
   - When triggered: fan chart overlays the stressed scenario in a different color (red/orange)
   - Delta summary card showing how default probability changed

6. **Summary Stats Bar** (always visible)
   - Median monthly income
   - Income volatility (CV)
   - Probability of any negative cash flow month
   - Months of runway (savings / avg monthly burn)

### 8.2 Key UX Requirements

- The fan chart is THE hero visualization. Make it large, clean, and beautiful.
- Animate the fan chart filling in when simulation runs (paths drawing left to right).
- Show a loading state with "Running 5,000 simulations..." while backend computes.
- The FICO comparison should be visually prominent. This is the insight that wins with judges.
- Mobile-friendly is NOT required (this is a hackathon demo on a laptop).
- Dark mode optional but looks better for demos.

### 8.3 Color Palette Suggestion

```
Primary:    #0066CC (Cap1-adjacent blue)
Secondary:  #00A878 (green for positive/safe)
Warning:    #FF6B35 (orange for caution)  
Danger:     #D32F2F (red for high risk)
Background: #0F172A (dark slate) or #FFFFFF (clean white)
Fan bands:  Blue gradient from 10% opacity (p10-p90) to 80% opacity (p50 line)
Stress overlay: Red/orange gradient at 40% opacity
```

---

## 9. BUILD TIMELINE (24 hours)

| Hours | Task | Owner |
|-------|------|-------|
| 0-2 | Scaffold Next.js + FastAPI, data files in place | Both |
| 2-6 | Monte Carlo engine core + `/api/simulate` endpoint | Backend |
| 2-6 | Profile input form + fan chart component + API integration | Frontend |
| 6-10 | Loan evaluation module + FICO comparison | Backend |
| 6-10 | Loan eval panel + summary stats + polish fan chart | Frontend |
| 10-14 | AI scenario agent + `/api/scenario` endpoint | Backend |
| 10-14 | Scenario panel + overlay on fan chart + preset buttons | Frontend |
| 14-18 | Compare endpoint (baseline vs stressed) + delta summary | Backend |
| 14-18 | Full integration, compare view, animation polish | Frontend |
| 18-22 | End-to-end testing, bug fixes, demo flow rehearsal | Both |
| 22-24 | Record demo video, final polish | Both |

### CUT LIST (if behind schedule)
1. Drop AI scenario agent entirely. Hardcode 3-4 preset scenarios as JSON.
2. Drop fan chart animation. Static chart is fine.
3. Drop compare view. Just show one simulation at a time.
4. Drop live FRED API calls. Use cached data only.

### STRETCH GOALS (if ahead)
1. "Loan Recommender" mode: system tests multiple loan structures and finds the one that minimizes default probability while maximizing borrower qualification
2. Historical backtest: "How would our model have scored this borrower in 2007?"
3. Income distribution histogram alongside the fan chart
4. Export simulation report as PDF

---

## 10. DEMO SCRIPT (for the pitch video)

1. "Meet Maria. She drives for Uber in DC, 30 hours a week. She wants a $15,000 auto loan."
2. Enter Maria's profile. Run simulation. Show the fan chart spreading out.
3. "FICO says she's a 660. That means 6% default risk. But watch what our simulation says..."
4. Show loan evaluation. Our model says 3.2% default risk. "FICO overestimates her risk."
5. "Why? Because Maria's income is volatile but predictable. She's been driving 18 months. She has $4,000 in savings. FICO doesn't see any of that."
6. "But what if things go wrong?" Type: "Gas prices hit $6/gallon for 6 months"
7. Show the stressed overlay. Default probability jumps to 8.1%. "Now she IS riskier than FICO thinks."
8. "This is the insight: gig workers aren't uniformly risky. Some are safer than FICO says. Some are more fragile. You need a simulation to tell them apart."
9. "We calibrated our model with real data from the JPMorgan Chase Institute and the Federal Reserve. 36% month-to-month income volatility. Seasonal spikes in March and December. Real gas prices. Real life shocks."

---

## 11. KEY SOURCES TO CITE

- JPMorgan Chase Institute, "Weathering Volatility 2.0" (2019) - income volatility parameters
- JPMorgan Chase Institute, "Paychecks, Paydays, and the Online Platform Economy" (2016) - gig worker income share
- JPMorgan Chase Institute, "Earnings Instability" - hourly worker earnings swings
- Gridwise (2025) - platform-specific earnings benchmarks
- Economic Policy Institute, "Uber and the Labor Market" - driver compensation analysis
- Berkeley Labor Center, "Gig Passenger and Delivery Driver Pay" (2024) - metro-level pay data
- Federal Reserve FRED database - macro economic indicators
- Bureau of Labor Statistics - occupational wage data

---

## 12. REPO STRUCTURE

```
gigsim/
├── README.md
├── SPEC.md                    # this file
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # main dashboard
│   │   └── api/               # Next.js API routes (proxy to backend)
│   ├── components/
│   │   ├── ProfileForm.tsx
│   │   ├── FanChart.tsx       # THE hero visualization
│   │   ├── LoanEvalPanel.tsx
│   │   ├── ScenarioPanel.tsx
│   │   ├── SummaryStats.tsx
│   │   └── FicoComparison.tsx
│   └── lib/
│       └── api.ts             # API client functions
├── backend/
│   ├── requirements.txt       # fastapi, uvicorn, numpy, httpx, pydantic
│   ├── main.py                # FastAPI app + routes
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── simulation.py      # Monte Carlo engine
│   │   ├── loan_evaluator.py  # Loan evaluation + FICO comparison
│   │   └── scenario_agent.py  # AI scenario translation
│   ├── data/
│   │   ├── calibration.py     # All hardcoded parameters from section 3.4
│   │   └── fred_client.py     # FRED API client with cache fallback
│   └── models/
│       └── schemas.py         # All Pydantic models
└── data/                      # Pre-cached JSON data files
    ├── fred/
    ├── calibration/
    └── scenarios/
```