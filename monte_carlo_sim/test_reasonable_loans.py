"""
Test archetypes with reasonable loan amounts (smaller, more appropriate).
This shows better differentiation by testing loans within each archetype's capacity.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.loaders import DataLoader
from src.engine.monte_carlo import load_and_prepare, run_simulation
from src.types import SimulationConfig, LoanConfig, WorkerProfile, GigStream, GigType

def create_archetype_profile(archetype: dict, loan_amount: float, term: int) -> WorkerProfile:
    """Create WorkerProfile directly from archetype with specified loan."""
    from src.types import GigStream, GigType
    
    gig_type_map = {
        "doordash": GigType.DELIVERY,
        "uber": GigType.RIDESHARE,
        "lyft": GigType.RIDESHARE,
        "instacart": GigType.DELIVERY,
        "grubhub": GigType.DELIVERY,
    }
    
    streams = []
    num_platforms = len(archetype["platforms"])
    income_per_stream = archetype["base_mu"] / num_platforms
    variance_per_stream = (archetype["base_sigma"] ** 2) / num_platforms
    
    for idx, platform in enumerate(archetype["platforms"]):
        gig_type = gig_type_map.get(platform.lower(), GigType.FREELANCE)
        streams.append(GigStream(
            platform_name=platform,
            gig_type=gig_type,
            mean_monthly_income=income_per_stream,
            income_variance=variance_per_stream,
            tenure_months=archetype["experience_months"],
            is_primary=(idx == 0),
        ))
    
    base_income = archetype["base_mu"]
    debt_ratio = archetype["debt_to_income_ratio"]
    emergency_weeks = archetype["emergency_fund_weeks"]
    
    monthly_debt = base_income * debt_ratio * 0.5
    monthly_expenses = base_income * 0.25
    liquid_savings = (base_income / 4.33) * emergency_weeks
    
    return WorkerProfile(
        streams=streams,
        metro_area=archetype["metro"],
        months_as_gig_worker=archetype["experience_months"],
        has_vehicle=True,
        has_dependents=False,
        liquid_savings=liquid_savings,
        monthly_fixed_expenses=monthly_expenses,
        existing_debt_obligations=monthly_debt,
        credit_score_range=tuple(archetype["credit_score_range"]),
        loan_request_amount=loan_amount,
        requested_term_months=term,
        acceptable_rate_range=(0.08, 0.24),
        correlation_matrix=None,
    )

def main():
    loader = DataLoader()
    config = SimulationConfig(n_paths=3000, horizon_months=24, random_seed=42)
    
    test_cases = [
        ('volatile_vic', 2000, 12, "High volatility, should be risky"),
        ('steady_sarah', 5000, 24, "Low volatility, should be safer"),
        ('weekend_warrior', 1500, 12, "Part-time, small loan, should be PRIME"),
        ('sf_hustler', 6000, 24, "High income, should handle moderate loan"),
        ('rising_ryan', 2500, 18, "Growing, moderate risk"),
    ]
    
    print("\n" + "="*80)
    print("REASONABLE LOAN AMOUNT TEST")
    print("="*80)
    print("\nTesting each archetype with appropriately sized loans")
    print("to verify the system differentiates based on risk profile.\n")
    
    results = []
    
    for arch_id, loan_amt, loan_term, description in test_cases:
        archetype = loader.load_archetype(arch_id)
        profile = create_archetype_profile(archetype, loan_amt, loan_term)
        load = load_and_prepare(profile, config)
        
        loan = LoanConfig(
            amount=loan_amt,
            term_months=loan_term,
            annual_rate=0.14
        )
        
        result = run_simulation(profile, config, loan, load, None)
        rec = result.recommended_loan
        
        print(f"\n{archetype['name']}:")
        print(f"  Profile: {description}")
        print(f"  Income: ${archetype['base_mu']:.0f}/mo (CV: {archetype['coefficient_of_variation']:.1%})")
        print(f"  Emergency: {archetype['emergency_fund_weeks']} weeks | Credit: {archetype['credit_score_range'][0]}-{archetype['credit_score_range'][1]}")
        print(f"  Loan: ${loan_amt:,} @ {loan.annual_rate:.0%} for {loan_term} months")
        print(f"  Result: P(def)={result.p_default:.2%} | Tier={rec.risk_tier.value.upper()} | Approved={rec.approved}")
        
        results.append({
            'name': archetype['name'],
            'income': archetype['base_mu'],
            'cv': archetype['coefficient_of_variation'],
            'loan': loan_amt,
            'p_default': result.p_default,
            'tier': rec.risk_tier.value,
            'approved': rec.approved,
        })
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(f"{'Archetype':<18} {'Income':<10} {'CV':<8} {'Loan':<10} {'P(def)':<10} {'Tier':<12} {'Status'}")
    print("-" * 80)
    for r in results:
        status = "APPROVED" if r['approved'] else "DECLINED"
        print(f"{r['name']:<18} ${r['income']:>7.0f}  {r['cv']:>6.1%}  ${r['loan']:>7,}  "
              f"{r['p_default']:>8.2%}  {r['tier']:<12} {status}")
    
    print(f"\n{'='*80}")
    print("DIFFERENTIATION ANALYSIS")
    print(f"{'='*80}")
    
    p_defaults = [r['p_default'] for r in results]
    tiers = [r['tier'] for r in results]
    approvals = [r['approved'] for r in results]
    
    print(f"\nP(default) spread: {min(p_defaults):.2%} to {max(p_defaults):.2%}")
    print(f"Range: {max(p_defaults) - min(p_defaults):.2%} percentage points")
    print(f"Risk tiers present: {set(tiers)}")
    print(f"Approval rate: {sum(approvals)}/{len(approvals)} ({sum(approvals)/len(approvals):.0%})")
    
    if len(set(tiers)) >= 3:
        print(f"\nPASS: System uses 3+ different risk tiers")
    if max(p_defaults) - min(p_defaults) > 0.1:
        print(f"PASS: P(default) varies by >10 percentage points")
    if sum(approvals) > 0 and sum(approvals) < len(approvals):
        print(f"PASS: Some approved, some declined (proper discrimination)")

if __name__ == "__main__":
    main()
