"""
Visualize Monte Carlo income paths and cash flow dynamics for detailed analysis.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from data_pipeline.loaders import DataLoader
from src.engine.monte_carlo import load_and_prepare, run_simulation
from src.types import SimulationConfig, LoanConfig, WorkerProfile, GigStream, GigType

def create_archetype_profile(archetype: dict, expenses_pct: float = 0.25, debt_pct: float = 0.5) -> WorkerProfile:
    """Create WorkerProfile directly from archetype with configurable expenses."""
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
    
    monthly_debt = base_income * debt_ratio * debt_pct
    monthly_expenses = base_income * expenses_pct
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

def visualize_comparison(arch1_id: str, arch2_id: str, loan_amount: float = 2000):
    """Compare two archetypes with detailed visualizations."""
    loader = DataLoader()
    
    arch1 = loader.load_archetype(arch1_id)
    arch2 = loader.load_archetype(arch2_id)
    
    profile1 = create_archetype_profile(arch1)
    profile2 = create_archetype_profile(arch2)
    
    config = SimulationConfig(n_paths=1000, horizon_months=24, random_seed=42)
    
    load1 = load_and_prepare(profile1, config)
    load2 = load_and_prepare(profile2, config)
    
    loan = LoanConfig(amount=loan_amount, term_months=24, annual_rate=0.14)
    
    result1 = run_simulation(profile1, config, loan, load1, None)
    result2 = run_simulation(profile2, config, loan, load2, None)
    
    print(f"\n{'='*80}")
    print(f"DETAILED COMPARISON: {arch1['name']} vs {arch2['name']}")
    print(f"{'='*80}\n")
    
    print(f"Loan: ${loan_amount:,} @ 14% for 24 months")
    print(f"Monthly Payment: ${loan.amount * 0.14 / 12 / (1 - (1 + 0.14/12)**-24):,.2f}\n")
    
    print(f"{'Metric':<30} {arch1['name']:<20} {arch2['name']:<20}")
    print("-" * 75)
    
    print(f"{'BASE ARCHETYPE DATA':<30}")
    print(f"  Mean Income (mu)             ${arch1['base_mu']:>17,.2f}  ${arch2['base_mu']:>17,.2f}")
    print(f"  Std Dev (sigma)              ${arch1['base_sigma']:>17,.2f}  ${arch2['base_sigma']:>17,.2f}")
    print(f"  CV                           {arch1['coefficient_of_variation']:>18.1%}  {arch2['coefficient_of_variation']:>18.1%}")
    print(f"  Platforms                    {len(arch1['platforms']):>18}  {len(arch2['platforms']):>18}")
    print(f"  Emergency Fund               {arch1['emergency_fund_weeks']:>15} wks  {arch2['emergency_fund_weeks']:>15} wks")
    print(f"  Credit Score                 {arch1['credit_score_range'][0]:>12}-{arch1['credit_score_range'][1]:<3}  {arch2['credit_score_range'][0]:>12}-{arch2['credit_score_range'][1]:<3}")
    
    total_income1 = sum(s.mean_monthly_income for s in profile1.streams)
    total_income2 = sum(s.mean_monthly_income for s in profile2.streams)
    total_exp1 = profile1.monthly_fixed_expenses + profile1.existing_debt_obligations
    total_exp2 = profile2.monthly_fixed_expenses + profile2.existing_debt_obligations
    
    print(f"\n{'PROFILE (USED IN SIM)':<30}")
    print(f"  Total Income                 ${total_income1:>17,.2f}  ${total_income2:>17,.2f}")
    print(f"  Fixed Expenses               ${profile1.monthly_fixed_expenses:>17,.2f}  ${profile2.monthly_fixed_expenses:>17,.2f}")
    print(f"  Existing Debt                ${profile1.existing_debt_obligations:>17,.2f}  ${profile2.existing_debt_obligations:>17,.2f}")
    print(f"  Total Obligations            ${total_exp1:>17,.2f}  ${total_exp2:>17,.2f}")
    print(f"  Free Cash Flow               ${total_income1 - total_exp1:>17,.2f}  ${total_income2 - total_exp2:>17,.2f}")
    print(f"  FCF %                        {(total_income1 - total_exp1)/total_income1:>18.1%}  {(total_income2 - total_exp2)/total_income2:>18.1%}")
    print(f"  Liquid Savings               ${profile1.liquid_savings:>17,.2f}  ${profile2.liquid_savings:>17,.2f}")
    print(f"  Savings / Income             {profile1.liquid_savings/total_income1:>18.2f}x {profile2.liquid_savings/total_income2:>18.2f}x")
    
    from src.engine.monte_carlo import _monthly_payment
    payment = _monthly_payment(loan_amount, 0.14, 24)
    fcf_after1 = total_income1 - total_exp1 - payment
    fcf_after2 = total_income2 - total_exp2 - payment
    
    print(f"\n{'AFTER LOAN PAYMENT':<30}")
    print(f"  Monthly Payment              ${payment:>17,.2f}  ${payment:>17,.2f}")
    print(f"  FCF After Payment            ${fcf_after1:>17,.2f}  ${fcf_after2:>17,.2f}")
    print(f"  Payment / Income             {payment/total_income1:>18.1%}  {payment/total_income2:>18.1%}")
    
    print(f"\n{'SIMULATION RESULTS':<30}")
    print(f"  P(default)                   {result1.p_default:>18.2%}  {result2.p_default:>18.2%}")
    print(f"  Expected Loss                ${result1.expected_loss:>17,.2f}  ${result2.expected_loss:>17,.2f}")
    print(f"  CVaR 95%                     ${result1.cvar_95:>17,.2f}  ${result2.cvar_95:>17,.2f}")
    print(f"  Tier                         {result1.recommended_loan.risk_tier.value:>18}  {result2.recommended_loan.risk_tier.value:>18}")
    print(f"  Approved                     {str(result1.recommended_loan.approved):>18}  {str(result2.recommended_loan.approved):>18}")
    
    print(f"\n{'INCOME PATH STATISTICS':<30}")
    print(f"  Median Income (P50)          ${np.median(result1.median_income_by_month):>17,.2f}  ${np.median(result2.median_income_by_month):>17,.2f}")
    print(f"  10th Percentile (P10)        ${np.median(result1.p10_income_by_month):>17,.2f}  ${np.median(result2.p10_income_by_month):>17,.2f}")
    print(f"  90th Percentile (P90)        ${np.median(result1.p90_income_by_month):>17,.2f}  ${np.median(result2.p90_income_by_month):>17,.2f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Comparison: {arch1['name']} vs {arch2['name']} (${loan_amount:,} loan)", 
                 fontsize=16, fontweight='bold')
    
    for idx, (result, profile, arch, name) in enumerate([
        (result1, profile1, arch1, arch1['name']),
        (result2, profile2, arch2, arch2['name'])
    ]):
        row = idx
        
        ax = axes[row, 0]
        paths_to_plot = min(100, config.n_paths)
        for i in range(paths_to_plot):
            ax.plot(result.raw_paths[i, :], alpha=0.1, color='blue' if idx == 0 else 'green')
        ax.plot(result.median_income_by_month, 'r-', linewidth=2, label='Median')
        ax.plot(result.p10_income_by_month, 'k--', linewidth=1, label='P10')
        ax.plot(result.p90_income_by_month, 'k--', linewidth=1, label='P90')
        ax.axhline(y=total_income1 if idx == 0 else total_income2, color='orange', 
                   linestyle=':', linewidth=2, label='Base mu')
        ax.set_title(f"{name}: Income Paths (CV={arch['coefficient_of_variation']:.1%})")
        ax.set_xlabel('Month')
        ax.set_ylabel('Income ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[row, 1]
        income_dist = result.raw_paths[:, 0]
        ax.hist(income_dist, bins=50, alpha=0.7, color='blue' if idx == 0 else 'green', edgecolor='black')
        ax.axvline(x=np.mean(income_dist), color='red', linewidth=2, label=f'Mean: ${np.mean(income_dist):,.0f}')
        ax.axvline(x=np.median(income_dist), color='orange', linewidth=2, label=f'Median: ${np.median(income_dist):,.0f}')
        ax.set_title(f"{name}: Income Distribution (Month 0)")
        ax.set_xlabel('Income ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[row, 2]
        total_exp = (profile1.monthly_fixed_expenses + profile1.existing_debt_obligations) if idx == 0 else \
                    (profile2.monthly_fixed_expenses + profile2.existing_debt_obligations)
        net_income = result.raw_paths - total_exp - payment
        for i in range(min(100, config.n_paths)):
            ax.plot(net_income[i, :], alpha=0.1, color='blue' if idx == 0 else 'green')
        ax.plot(np.median(net_income, axis=0), 'r-', linewidth=2, label='Median Net')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=-payment, color='red', linestyle='--', linewidth=2, label=f'Payment: ${payment:.0f}')
        ax.set_title(f"{name}: Net Cash Flow After Expenses + Loan")
        ax.set_xlabel('Month')
        ax.set_ylabel('Net Cash Flow ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = ROOT / f"comparison_{arch1_id}_vs_{arch2_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

def detailed_profile_analysis(arch_id: str, loan_amount: float = 2000):
    """Deep dive analysis of a single archetype."""
    loader = DataLoader()
    archetype = loader.load_archetype(arch_id)
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {archetype['name']}")
    print(f"{'='*80}\n")
    
    profile = create_archetype_profile(archetype)
    
    print("ARCHETYPE DATA (from data_pipeline):")
    print(f"  base_mu: ${archetype['base_mu']:,.2f}")
    print(f"  base_sigma: ${archetype['base_sigma']:,.2f}")
    print(f"  CV: {archetype['coefficient_of_variation']:.1%}")
    print(f"  Platforms: {archetype['platforms']}")
    print(f"  Hours/week: {archetype['hours_per_week']}")
    print(f"  Metro: {archetype['metro']}")
    print(f"  Emergency fund: {archetype['emergency_fund_weeks']} weeks")
    print(f"  Debt-to-income: {archetype['debt_to_income_ratio']:.1%}")
    print(f"  Credit score: {archetype['credit_score_range']}")
    
    print(f"\nWORKERPROFILE CONSTRUCTED:")
    print(f"  Number of streams: {len(profile.streams)}")
    for i, stream in enumerate(profile.streams):
        stream_mu = stream.mean_monthly_income
        stream_sigma = np.sqrt(stream.income_variance)
        stream_cv = stream_sigma / stream_mu if stream_mu > 0 else 0
        print(f"    Stream {i+1} ({stream.platform_name}):")
        print(f"      mu = ${stream_mu:,.2f}, sigma = ${stream_sigma:,.2f}, CV = {stream_cv:.1%}")
    
    total_income = sum(s.mean_monthly_income for s in profile.streams)
    total_var = sum(s.income_variance for s in profile.streams)
    portfolio_sigma = np.sqrt(total_var)
    portfolio_cv = portfolio_sigma / total_income if total_income > 0 else 0
    
    print(f"\n  Portfolio totals:")
    print(f"    Total mu: ${total_income:,.2f}")
    print(f"    Total sigma: ${portfolio_sigma:,.2f}")
    print(f"    Portfolio CV: {portfolio_cv:.1%}")
    
    print(f"\n  Monthly obligations:")
    print(f"    Fixed expenses: ${profile.monthly_fixed_expenses:,.2f} ({profile.monthly_fixed_expenses/total_income:.1%} of income)")
    print(f"    Existing debt: ${profile.existing_debt_obligations:,.2f} ({profile.existing_debt_obligations/total_income:.1%} of income)")
    print(f"    Total: ${profile.monthly_fixed_expenses + profile.existing_debt_obligations:,.2f}")
    
    print(f"\n  Liquid savings: ${profile.liquid_savings:,.2f} ({profile.liquid_savings/total_income:.2f}x monthly income)")
    
    total_exp = profile.monthly_fixed_expenses + profile.existing_debt_obligations
    fcf = total_income - total_exp
    print(f"\n  Free cash flow BEFORE loan: ${fcf:,.2f} ({fcf/total_income:.1%} of income)")
    
    from src.engine.monte_carlo import _monthly_payment
    payment = _monthly_payment(loan_amount, 0.14, 24)
    fcf_after = fcf - payment
    
    print(f"\n  Loan payment: ${payment:,.2f} ({payment/total_income:.1%} of income)")
    print(f"  FCF AFTER loan: ${fcf_after:,.2f} ({fcf_after/total_income:.1%} of income)")
    
    loan = LoanConfig(amount=loan_amount, term_months=24, annual_rate=0.14)
    config_full = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=42)
    load = load_and_prepare(profile, config_full)
    result = run_simulation(profile, config_full, loan, load, None)
    
    print(f"\nSIMULATION RESULTS (5000 paths):")
    print(f"  P(default): {result.p_default:.2%}")
    print(f"  Expected loss: ${result.expected_loss:,.2f}")
    print(f"  CVaR 95%: ${result.cvar_95:,.2f}")
    print(f"  Risk tier: {result.recommended_loan.risk_tier.value.upper()}")
    print(f"  Approved: {result.recommended_loan.approved}")
    
    print(f"\nINCOME PATH PERCENTILES:")
    print(f"  P10 (worst 10%): ${np.median(result.p10_income_by_month):,.2f}")
    print(f"  P50 (median): ${np.median(result.median_income_by_month):,.2f}")
    print(f"  P90 (best 10%): ${np.median(result.p90_income_by_month):,.2f}")
    
    worst_paths = np.argsort(np.min(result.raw_paths, axis=1))[:10]
    print(f"\nWORST 10 PATHS (min income during horizon):")
    for i in worst_paths:
        min_income = np.min(result.raw_paths[i, :])
        print(f"  Path {i}: Min income = ${min_income:,.2f} (month {np.argmin(result.raw_paths[i, :])})")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{archetype['name']} - ${loan_amount:,} Loan Analysis", fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    paths_to_plot = min(200, config_full.n_paths)
    for i in range(paths_to_plot):
        ax.plot(result.raw_paths[i, :], alpha=0.05, color='steelblue')
    ax.plot(result.median_income_by_month, 'r-', linewidth=2, label='Median', zorder=10)
    ax.plot(result.p10_income_by_month, 'orange', linewidth=2, label='P10 (worst 10%)', zorder=10)
    ax.plot(result.p90_income_by_month, 'green', linewidth=2, label='P90 (best 10%)', zorder=10)
    ax.axhline(y=total_income, color='black', linestyle='--', linewidth=1, label='Base mu', alpha=0.7)
    ax.axhline(y=total_exp + payment, color='red', linestyle=':', linewidth=2, 
               label=f'Total Obligations+Payment: ${total_exp + payment:.0f}', alpha=0.8)
    ax.set_title('Income Paths Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Income ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    month0_income = result.raw_paths[:, 0]
    ax.hist(month0_income, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=np.mean(month0_income), color='red', linewidth=2, label=f'Mean: ${np.mean(month0_income):,.0f}')
    ax.axvline(x=np.percentile(month0_income, 10), color='orange', linewidth=2, 
               linestyle='--', label=f'P10: ${np.percentile(month0_income, 10):,.0f}')
    ax.axvline(x=total_exp + payment, color='red', linestyle=':', linewidth=2, 
               label=f'Need: ${total_exp + payment:.0f}')
    ax.set_title('Income Distribution (Month 0)')
    ax.set_xlabel('Income ($)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    net_cf = result.raw_paths - (total_exp + payment)
    for i in range(min(200, config_full.n_paths)):
        ax.plot(net_cf[i, :], alpha=0.05, color='steelblue')
    ax.plot(np.median(net_cf, axis=0), 'r-', linewidth=2, label='Median Net CF')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.fill_between(range(24), 0, -payment, alpha=0.2, color='red', label='Danger Zone')
    ax.set_title('Net Cash Flow (Income - Expenses - Payment)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Cash Flow ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    min_income_per_path = np.min(result.raw_paths, axis=1)
    ax.hist(min_income_per_path, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=total_exp + payment, color='red', linewidth=2, linestyle=':', 
               label=f'Survival Threshold: ${total_exp + payment:.0f}')
    paths_below = np.sum(min_income_per_path < total_exp + payment)
    ax.set_title(f'Minimum Income Per Path\n({paths_below} paths drop below threshold)')
    ax.set_xlabel('Minimum Income During 24 Months ($)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = ROOT / f"analysis_{arch_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("MONTE CARLO PATH VISUALIZATION AND ANALYSIS")
    print("="*80)
    
    print("\n[1] Comparing Steady Sarah vs Weekend Warrior")
    visualize_comparison('steady_sarah', 'weekend_warrior', loan_amount=2000)
    
    print("\n" + "="*80)
    print("\n[2] Detailed Analysis: Steady Sarah")
    detailed_profile_analysis('steady_sarah', loan_amount=2000)
    
    print("\n" + "="*80)
    print("\n[3] Detailed Analysis: Volatile Vic")
    detailed_profile_analysis('volatile_vic', loan_amount=2000)
    
    print("\n" + "="*80)
    print("\n[4] Detailed Analysis: Weekend Warrior")
    detailed_profile_analysis('weekend_warrior', loan_amount=1500)
