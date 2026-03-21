"""
Test suite using 5 pre-built archetypes to verify risk differentiation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.loaders import DataLoader
from src.engine.monte_carlo import load_and_prepare, run_simulation, sweep_loan_space
from src.types import SimulationConfig, LoanConfig, WorkerProfile


def archetype_to_worker_profile(archetype: dict) -> WorkerProfile:
    """
    Convert archetype directly to WorkerProfile using archetype's pre-calculated mu/sigma.
    This bypasses profile_builder to use the archetype's exact parameters.
    """
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
    
    loan_min, loan_max = archetype["recommended_loan_amount_range"]
    loan_request = (loan_min + loan_max) / 2
    
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
        loan_request_amount=loan_request,
        requested_term_months=archetype["recommended_loan_term_months"],
        acceptable_rate_range=(0.08, 0.24),
        correlation_matrix=None,
    )

def test_archetype(archetype_id: str, loader: DataLoader):
    """Run simulation for a specific archetype."""
    archetype = loader.load_archetype(archetype_id)
    
    print(f"\n{'='*80}")
    print(f"Testing: {archetype['name']}")
    print(f"{'='*80}")
    print(f"Description: {archetype['description']}")
    print(f"Base Income: ${archetype['base_mu']:.2f}/mo (CV: {archetype['coefficient_of_variation']:.1%})")
    print(f"Platforms: {', '.join(archetype['platforms'])}")
    print(f"Emergency Fund: {archetype['emergency_fund_weeks']} weeks")
    print(f"Credit Score: {archetype['credit_score_range'][0]}-{archetype['credit_score_range'][1]}")
    print(f"Expected Risk: {archetype['default_risk_category'].upper()}")
    
    profile = archetype_to_worker_profile(archetype)
    
    config = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=42)
    load = load_and_prepare(profile, config)
    
    loan = LoanConfig(
        amount=profile.loan_request_amount,
        term_months=profile.requested_term_months,
        annual_rate=0.15
    )
    
    result = run_simulation(profile, config, loan, load, None)
    rec = result.recommended_loan
    
    print(f"\nRequested Loan: ${loan.amount:,.0f} @ {loan.annual_rate:.0%} for {loan.term_months} months")
    print(f"\nSimulation Results:")
    print(f"  P(default): {result.p_default:.2%}")
    print(f"  Expected Loss: ${result.expected_loss:,.2f}")
    print(f"  CVaR 95%: ${result.cvar_95:,.2f}")
    print(f"  Risk Tier: {rec.risk_tier.value.upper()}")
    print(f"  Approved: {'YES' if rec.approved else 'NO'}")
    
    print(f"\nRunning loan sweep to find optimal structure...")
    grid, optimal = sweep_loan_space(profile, config, load, None)
    
    if optimal:
        print(f"\nOptimal Loan Found:")
        print(f"  Amount: ${optimal['amount']:,.0f}")
        print(f"  Term: {optimal['term_months']} months")
        print(f"  Rate: {optimal['annual_rate']:.1%}")
        print(f"  P(default): {optimal['p_default']:.2%}")
    else:
        print(f"\nWARNING: NO LOAN MEETS APPROVAL THRESHOLD - Recommend smaller amount")
    
    return {
        'archetype': archetype['name'],
        'requested_amount': loan.amount,
        'p_default': result.p_default,
        'tier': rec.risk_tier.value,
        'approved': rec.approved,
        'optimal_amount': optimal['amount'] if optimal else 0,
        'optimal_rate': optimal['annual_rate'] if optimal else 0,
    }

def main():
    print("="*80)
    print("ARCHETYPE DIFFERENTIATION TEST SUITE")
    print("="*80)
    print("\nThis test verifies the system produces different outcomes for")
    print("borrowers with different risk profiles.\n")
    
    loader = DataLoader()
    archetype_ids = loader.list_archetypes()
    
    results = []
    for arch_id in archetype_ids:
        result = test_archetype(arch_id, loader)
        results.append(result)
    
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Archetype':<20} {'P(default)':<12} {'Tier':<12} {'Approved':<10} {'Optimal $':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['archetype']:<20} {r['p_default']:>10.2%}  {r['tier']:<12} "
              f"{'YES' if r['approved'] else 'NO':<10} ${r['optimal_amount']:>10,.0f}")
    
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")
    
    p_defaults = [r['p_default'] for r in results]
    optimal_amounts = [r['optimal_amount'] for r in results]
    tiers = [r['tier'] for r in results]
    
    print(f"\nP(default) range: {min(p_defaults):.2%} to {max(p_defaults):.2%}")
    print(f"Optimal loan range: ${min(optimal_amounts):,.0f} to ${max(optimal_amounts):,.0f}")
    print(f"Risk tiers used: {set(tiers)}")
    
    if len(set(optimal_amounts)) == 1 and optimal_amounts[0] > 0:
        print(f"\nWARNING: All archetypes got same optimal loan (${optimal_amounts[0]:,.0f})")
        print("   This suggests the system is not differentiating properly.")
    elif len(set(optimal_amounts)) >= 3:
        print(f"\nPASS: System shows good differentiation across archetypes")
    
if __name__ == "__main__":
    main()
