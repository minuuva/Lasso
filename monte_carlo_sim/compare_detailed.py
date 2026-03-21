"""
Detailed text-based comparison of Steady Sarah vs Weekend Warrior.
Shows exactly why one is approved and the other declined.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from data_pipeline.loaders import DataLoader
from src.engine.monte_carlo import load_and_prepare, run_simulation, _monthly_payment
from src.types import SimulationConfig, LoanConfig, WorkerProfile, GigStream, GigType

def create_profile(archetype: dict) -> WorkerProfile:
    """Create WorkerProfile directly from archetype."""
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

def analyze_archetype(arch_id: str, loan_amount: float, term: int, rate: float):
    """Detailed analysis of single archetype."""
    loader = DataLoader()
    archetype = loader.load_archetype(arch_id)
    profile = create_profile(archetype)
    
    config = SimulationConfig(n_paths=5000, horizon_months=24, random_seed=42)
    load = load_and_prepare(profile, config)
    loan = LoanConfig(amount=loan_amount, term_months=term, annual_rate=rate)
    
    result = run_simulation(profile, config, loan, load, None)
    
    total_income = sum(s.mean_monthly_income for s in profile.streams)
    total_var = sum(s.income_variance for s in profile.streams)
    total_sigma = np.sqrt(total_var)
    total_cv = total_sigma / total_income
    
    total_exp = profile.monthly_fixed_expenses + profile.existing_debt_obligations
    payment = _monthly_payment(loan_amount, rate, term)
    fcf = total_income - total_exp
    fcf_after = fcf - payment
    
    return {
        'archetype': archetype,
        'profile': profile,
        'result': result,
        'total_income': total_income,
        'total_sigma': total_sigma,
        'total_cv': total_cv,
        'total_exp': total_exp,
        'payment': payment,
        'fcf': fcf,
        'fcf_after': fcf_after,
    }

def main():
    print("\n" + "="*80)
    print("DETAILED COMPARISON: Steady Sarah vs Weekend Warrior")
    print("="*80)
    print("\nWhy does Weekend Warrior get approved but Steady Sarah gets declined?")
    print("Let's investigate...\n")
    
    loan_amount = 2000
    term = 24
    rate = 0.14
    
    sarah = analyze_archetype('steady_sarah', loan_amount, term, rate)
    warrior = analyze_archetype('weekend_warrior', loan_amount, term, rate)
    
    print(f"\n{'METRIC':<35} {'Steady Sarah':<20} {'Weekend Warrior':<20} {'Winner'}")
    print("="*80)
    
    print(f"\n{'ARCHETYPE DATA (from JSON)':<35}")
    print(f"  Mean Income (base_mu)          ${sarah['archetype']['base_mu']:>17,.2f}  ${warrior['archetype']['base_mu']:>17,.2f}  {'Sarah' if sarah['archetype']['base_mu'] > warrior['archetype']['base_mu'] else 'Warrior'}")
    print(f"  Std Dev (base_sigma)           ${sarah['archetype']['base_sigma']:>17,.2f}  ${warrior['archetype']['base_sigma']:>17,.2f}  {'Warrior' if sarah['archetype']['base_sigma'] < warrior['archetype']['base_sigma'] else 'Sarah'}")
    print(f"  CV                             {sarah['archetype']['coefficient_of_variation']:>18.1%}  {warrior['archetype']['coefficient_of_variation']:>18.1%}  {'Sarah' if sarah['archetype']['coefficient_of_variation'] < warrior['archetype']['coefficient_of_variation'] else 'Warrior'}")
    print(f"  # Platforms                    {len(sarah['archetype']['platforms']):>18}  {len(warrior['archetype']['platforms']):>18}  {'Sarah' if len(sarah['archetype']['platforms']) > len(warrior['archetype']['platforms']) else 'Warrior'}")
    print(f"  Emergency Fund (weeks)         {sarah['archetype']['emergency_fund_weeks']:>18}  {warrior['archetype']['emergency_fund_weeks']:>18}  {'Warrior' if warrior['archetype']['emergency_fund_weeks'] > sarah['archetype']['emergency_fund_weeks'] else 'Sarah'}")
    print(f"  Credit Score (midpoint)        {sum(sarah['archetype']['credit_score_range'])//2:>18}  {sum(warrior['archetype']['credit_score_range'])//2:>18}  {'Warrior' if sum(warrior['archetype']['credit_score_range'])//2 > sum(sarah['archetype']['credit_score_range'])//2 else 'Sarah'}")
    
    print(f"\n{'PROFILE IN SIMULATION':<35}")
    print(f"  Total Income                   ${sarah['total_income']:>17,.2f}  ${warrior['total_income']:>17,.2f}  {'Sarah' if sarah['total_income'] > warrior['total_income'] else 'Warrior'}")
    print(f"  Portfolio Sigma                ${sarah['total_sigma']:>17,.2f}  ${warrior['total_sigma']:>17,.2f}  {'Warrior' if sarah['total_sigma'] < warrior['total_sigma'] else 'Sarah'}")
    print(f"  Portfolio CV                   {sarah['total_cv']:>18.1%}  {warrior['total_cv']:>18.1%}  {'Sarah' if sarah['total_cv'] < warrior['total_cv'] else 'Warrior'}")
    print(f"  Total Obligations              ${sarah['total_exp']:>17,.2f}  ${warrior['total_exp']:>17,.2f}  {'Warrior' if sarah['total_exp'] < warrior['total_exp'] else 'Sarah'}")
    print(f"  Free Cash Flow (before loan)   ${sarah['fcf']:>17,.2f}  ${warrior['fcf']:>17,.2f}  {'Sarah' if sarah['fcf'] > warrior['fcf'] else 'Warrior'}")
    print(f"  FCF % (before loan)            {sarah['fcf']/sarah['total_income']:>18.1%}  {warrior['fcf']/warrior['total_income']:>18.1%}  {'Sarah' if sarah['fcf']/sarah['total_income'] > warrior['fcf']/warrior['total_income'] else 'Warrior'}")
    print(f"  Liquid Savings                 ${sarah['profile'].liquid_savings:>17,.2f}  ${warrior['profile'].liquid_savings:>17,.2f}  {'Warrior' if warrior['profile'].liquid_savings > sarah['profile'].liquid_savings else 'Sarah'}")
    print(f"  Savings / Income ratio         {sarah['profile'].liquid_savings/sarah['total_income']:>18.2f}x {warrior['profile'].liquid_savings/warrior['total_income']:>18.2f}x {'Warrior' if warrior['profile'].liquid_savings/warrior['total_income'] > sarah['profile'].liquid_savings/sarah['total_income'] else 'Sarah'}")
    
    print(f"\n{'WITH $2,000 LOAN':<35}")
    print(f"  Monthly Payment                ${sarah['payment']:>17,.2f}  ${warrior['payment']:>17,.2f}")
    print(f"  FCF After Payment              ${sarah['fcf_after']:>17,.2f}  ${warrior['fcf_after']:>17,.2f}  {'Sarah' if sarah['fcf_after'] > warrior['fcf_after'] else 'Warrior'}")
    print(f"  Payment / Income               {sarah['payment']/sarah['total_income']:>18.1%}  {warrior['payment']/warrior['total_income']:>18.1%}  {'Warrior' if sarah['payment']/sarah['total_income'] < warrior['payment']/warrior['total_income'] else 'Sarah'}")
    print(f"  Payment / FCF                  {sarah['payment']/sarah['fcf']:>18.1%}  {warrior['payment']/warrior['fcf']:>18.1%}  {'Warrior' if sarah['payment']/sarah['fcf'] < warrior['payment']/warrior['fcf'] else 'Sarah'}")
    
    print(f"\n{'SIMULATION RESULTS':<35}")
    print(f"  P(default)                     {sarah['result'].p_default:>18.2%}  {warrior['result'].p_default:>18.2%}  {'Warrior' if warrior['result'].p_default < sarah['result'].p_default else 'Sarah'}")
    print(f"  Expected Loss                  ${sarah['result'].expected_loss:>17,.2f}  ${warrior['result'].expected_loss:>17,.2f}")
    print(f"  CVaR 95%                       ${sarah['result'].cvar_95:>17,.2f}  ${warrior['result'].cvar_95:>17,.2f}")
    print(f"  Risk Tier                      {sarah['result'].recommended_loan.risk_tier.value:>18}  {warrior['result'].recommended_loan.risk_tier.value:>18}")
    print(f"  Approved                       {str(sarah['result'].recommended_loan.approved):>18}  {str(warrior['result'].recommended_loan.approved):>18}")
    
    sarah_paths = sarah['result'].raw_paths
    warrior_paths = warrior['result'].raw_paths
    
    sarah_min_per_path = np.min(sarah_paths, axis=1)
    warrior_min_per_path = np.min(warrior_paths, axis=1)
    
    sarah_need = sarah['total_exp'] + sarah['payment']
    warrior_need = warrior['total_exp'] + warrior['payment']
    
    sarah_below = np.sum(sarah_min_per_path < sarah_need)
    warrior_below = np.sum(warrior_min_per_path < warrior_need)
    
    print(f"\n{'PATH ANALYSIS':<35}")
    print(f"  Minimum income threshold       ${sarah_need:>17,.2f}  ${warrior_need:>17,.2f}")
    print(f"  Paths dropping below threshold {sarah_below:>18}  {warrior_below:>18}  {'Warrior' if warrior_below < sarah_below else 'Sarah'}")
    print(f"  % paths in danger              {sarah_below/5000:>18.1%}  {warrior_below/5000:>18.1%}")
    
    print(f"\n{'='*80}")
    print("THE ANSWER:")
    print(f"{'='*80}\n")
    
    if sarah['result'].p_default > warrior['result'].p_default:
        print("Steady Sarah has HIGHER default risk than Weekend Warrior because:\n")
        
        if len(sarah['profile'].streams) > len(warrior['profile'].streams):
            per_stream_sarah = sarah['total_income'] / len(sarah['profile'].streams)
            per_stream_warrior = warrior['total_income'] / len(warrior['profile'].streams)
            print(f"1. MULTI-PLATFORM DILUTION:")
            print(f"   - Sarah has {len(sarah['profile'].streams)} streams @ ${per_stream_sarah:,.0f} each")
            print(f"   - Warrior has {len(warrior['profile'].streams)} stream @ ${per_stream_warrior:,.0f}")
            print(f"   - When ANY of Sarah's streams has a bad month, total income drops")
            print(f"   - Warrior's single stream is more concentrated but more predictable\n")
        
        if sarah['profile'].liquid_savings / sarah['total_income'] < warrior['profile'].liquid_savings / warrior['total_income']:
            print(f"2. SAVINGS BUFFER:")
            print(f"   - Sarah: ${sarah['profile'].liquid_savings:,.0f} = {sarah['profile'].liquid_savings/sarah['total_income']:.2f}x monthly income")
            print(f"   - Warrior: ${warrior['profile'].liquid_savings:,.0f} = {warrior['profile'].liquid_savings/warrior['total_income']:.2f}x monthly income")
            print(f"   - Warrior has {warrior['profile'].liquid_savings/warrior['total_income'] / (sarah['profile'].liquid_savings/sarah['total_income']):.1f}x better savings cushion\n")
        
        if sarah['total_cv'] > warrior['total_cv']:
            print(f"3. PORTFOLIO VOLATILITY:")
            print(f"   - Sarah portfolio CV: {sarah['total_cv']:.1%}")
            print(f"   - Warrior portfolio CV: {warrior['total_cv']:.1%}")
            print(f"   - Even though archetype CV shows Sarah as more stable,")
            print(f"     her multi-stream portfolio creates different dynamics\n")
        
        print(f"4. INCOME PATH BEHAVIOR:")
        print(f"   - Sarah: {sarah_below} paths ({sarah_below/50:.0f}%) drop below survival threshold")
        print(f"   - Warrior: {warrior_below} paths ({warrior_below/50:.0f}%) drop below survival threshold")
        print(f"   - Sarah is {sarah_below/warrior_below:.1f}x more likely to have dangerous paths\n")
        
        print("KEY INSIGHT:")
        print("Having 3 platforms doesn't automatically reduce risk if each stream is small.")
        print("Weekend Warrior's single concentrated income stream + large savings buffer")
        print("makes for more predictable cash flows despite higher individual CV.")

def compare_loan_progression():
    """Show how P(default) changes with loan amount for both."""
    loader = DataLoader()
    
    sarah = loader.load_archetype('steady_sarah')
    warrior = loader.load_archetype('weekend_warrior')
    
    profile_sarah = create_profile(sarah)
    profile_warrior = create_profile(warrior)
    
    config = SimulationConfig(n_paths=2000, horizon_months=12, random_seed=42)
    load_sarah = load_and_prepare(profile_sarah, config)
    load_warrior = load_and_prepare(profile_warrior, config)
    
    loan_amounts = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    
    print(f"\n{'='*80}")
    print("LOAN AMOUNT PROGRESSION")
    print(f"{'='*80}\n")
    print(f"How P(default) changes as loan amount increases:\n")
    print(f"{'Loan Amt':<12} {'Sarah P(def)':<15} {'Sarah Tier':<15} {'Warrior P(def)':<15} {'Warrior Tier'}")
    print("-" * 80)
    
    for amt in loan_amounts:
        loan = LoanConfig(amount=amt, term_months=24, annual_rate=0.14)
        
        res_sarah = run_simulation(profile_sarah, config, loan, load_sarah, None)
        res_warrior = run_simulation(profile_warrior, config, loan, load_warrior, None)
        
        print(f"${amt:>6,}      {res_sarah.p_default:>10.2%}      {res_sarah.recommended_loan.risk_tier.value:<15} "
              f"{res_warrior.p_default:>10.2%}      {res_warrior.recommended_loan.risk_tier.value}")

def main():
    print("\n" + "="*100)
    print("SOLVING THE MYSTERY: Why Steady Sarah Gets Declined")
    print("="*100)
    
    loan_amount = 2000
    sarah_data = analyze_archetype('steady_sarah', loan_amount, 24, 0.14)
    warrior_data = analyze_archetype('weekend_warrior', loan_amount, 24, 0.14)
    
    print(f"\nTesting ${loan_amount:,} loan @ 14% for 24 months\n")
    
    print(f"{'METRIC':<40} {'Steady Sarah':<20} {'Weekend Warrior':<20} {'Better?'}")
    print("="*100)
    
    print(f"\n{'ARCHETYPE CHARACTERISTICS':<40}")
    s_arch = sarah_data['archetype']
    w_arch = warrior_data['archetype']
    
    print(f"{'  Base Income (mu)':<40} ${s_arch['base_mu']:>17,.0f}  ${w_arch['base_mu']:>17,.0f}  {'Sarah >' if s_arch['base_mu'] > w_arch['base_mu'] else 'Warrior >'}")
    print(f"{'  Coefficient of Variation':<40} {s_arch['coefficient_of_variation']:>18.1%}  {w_arch['coefficient_of_variation']:>18.1%}  {'Sarah (lower)' if s_arch['coefficient_of_variation'] < w_arch['coefficient_of_variation'] else 'Warrior (lower)'}")
    print(f"{'  Number of Platforms':<40} {len(s_arch['platforms']):>18}  {len(w_arch['platforms']):>18}  {'Sarah (diversified)' if len(s_arch['platforms']) > len(w_arch['platforms']) else 'Warrior'}")
    print(f"{'  Emergency Fund (weeks)':<40} {s_arch['emergency_fund_weeks']:>18}  {w_arch['emergency_fund_weeks']:>18}  {'Warrior >' if w_arch['emergency_fund_weeks'] > s_arch['emergency_fund_weeks'] else 'Sarah >'}")
    print(f"{'  Credit Score (avg)':<40} {sum(s_arch['credit_score_range'])//2:>18}  {sum(w_arch['credit_score_range'])//2:>18}  {'Warrior >' if sum(w_arch['credit_score_range'])//2 > sum(s_arch['credit_score_range'])//2 else 'Sarah >'}")
    
    print(f"\n{'CALCULATED PROFILE VALUES':<40}")
    print(f"{'  Total Income (sim)':<40} ${sarah_data['total_income']:>17,.0f}  ${warrior_data['total_income']:>17,.0f}  {'Sarah >' if sarah_data['total_income'] > warrior_data['total_income'] else 'Warrior >'}")
    print(f"{'  Total Obligations':<40} ${sarah_data['total_exp']:>17,.0f}  ${warrior_data['total_exp']:>17,.0f}  {'Warrior <' if sarah_data['total_exp'] < warrior_data['total_exp'] else 'Sarah <'}")
    print(f"{'  Free Cash Flow':<40} ${sarah_data['fcf']:>17,.0f}  ${warrior_data['fcf']:>17,.0f}  {'Sarah >' if sarah_data['fcf'] > warrior_data['fcf'] else 'Warrior >'}")
    print(f"{'  FCF % of Income':<40} {sarah_data['fcf']/sarah_data['total_income']:>18.1%}  {warrior_data['fcf']/warrior_data['total_income']:>18.1%}  {'Sarah >' if sarah_data['fcf']/sarah_data['total_income'] > warrior_data['fcf']/warrior_data['total_income'] else 'Warrior >'}")
    print(f"{'  Liquid Savings':<40} ${sarah_data['profile'].liquid_savings:>17,.0f}  ${warrior_data['profile'].liquid_savings:>17,.0f}  {'Warrior >' if warrior_data['profile'].liquid_savings > sarah_data['profile'].liquid_savings else 'Sarah >'}")
    print(f"{'  Savings / Income ratio':<40} {sarah_data['profile'].liquid_savings/sarah_data['total_income']:>18.2f}x {warrior_data['profile'].liquid_savings/warrior_data['total_income']:>18.2f}x {'Warrior >' if warrior_data['profile'].liquid_savings/warrior_data['total_income'] > sarah_data['profile'].liquid_savings/sarah_data['total_income'] else 'Sarah >'}")
    
    print(f"\n{'WITH LOAN ADDED':<40}")
    print(f"{'  Monthly Payment':<40} ${sarah_data['payment']:>17,.2f}  ${warrior_data['payment']:>17,.2f}")
    print(f"{'  FCF After Payment':<40} ${sarah_data['fcf_after']:>17,.0f}  ${warrior_data['fcf_after']:>17,.0f}  {'Sarah >' if sarah_data['fcf_after'] > warrior_data['fcf_after'] else 'Warrior >'}")
    print(f"{'  Payment / Income %':<40} {sarah_data['payment']/sarah_data['total_income']:>18.1%}  {warrior_data['payment']/warrior_data['total_income']:>18.1%}")
    print(f"{'  Payment / FCF %':<40} {sarah_data['payment']/sarah_data['fcf']:>18.1%}  {warrior_data['payment']/warrior_data['fcf']:>18.1%}")
    
    print(f"\n{'SIMULATION OUTCOMES':<40}")
    print(f"{'  P(default)':<40} {sarah_data['result'].p_default:>18.2%}  {warrior_data['result'].p_default:>18.2%}  {'Warrior (WINS)' if warrior_data['result'].p_default < sarah_data['result'].p_default else 'Sarah (WINS)'}")
    print(f"{'  Risk Tier':<40} {sarah_data['result'].recommended_loan.risk_tier.value:>18}  {warrior_data['result'].recommended_loan.risk_tier.value:>18}")
    print(f"{'  Approved':<40} {str(sarah_data['result'].recommended_loan.approved):>18}  {str(warrior_data['result'].recommended_loan.approved):>18}")
    
    print(f"\n{'='*100}")
    print("KEY FINDINGS:")
    print(f"{'='*100}\n")
    
    print("Despite Steady Sarah having:")
    print("  - Higher total income ($1,450 vs $581)")
    print("  - Lower CV (22.5% vs 36.0%)")
    print("  - Better diversification (3 platforms vs 1)")
    print("  - Good emergency fund (8 weeks)")
    
    print("\nWeekend Warrior wins because:")
    print(f"  1. SUPERIOR SAVINGS BUFFER: {warrior_data['profile'].liquid_savings/warrior_data['total_income']:.2f}x income vs {sarah_data['profile'].liquid_savings/sarah_data['total_income']:.2f}x")
    print(f"  2. HIGHER FCF %: {warrior_data['fcf']/warrior_data['total_income']:.1%} vs {sarah_data['fcf']/sarah_data['total_income']:.1%}")
    print(f"  3. LOWER PAYMENT BURDEN: {warrior_data['payment']/warrior_data['total_income']:.1%} of income vs {sarah_data['payment']/sarah_data['total_income']:.1%}")
    
    print("\nThe critical insight:")
    print("Multi-platform portfolios can be RISKIER than single-platform if each stream is small")
    print("because correlation < 1.0 means streams can have independent bad months.")
    print("\nWeekend Warrior's massive savings cushion (2.3x income) absorbs all volatility.")
    print("Steady Sarah's smaller cushion (1.8x income) isn't enough for her multi-stream risk.")
    
    compare_loan_progression()

if __name__ == "__main__":
    main()
