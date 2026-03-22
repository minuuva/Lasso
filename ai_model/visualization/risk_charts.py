"""
Risk Charts - Specialized visualizations for risk metrics.

P(default) surfaces, CVaR evolution, and default timing analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Optional, List, Tuple


def plot_default_timing_analysis(
    result,
    archetype: dict,
    run_id: str = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> Path:
    """
    Analyze when defaults occur across all paths.
    
    Args:
        result: SimulationResult object
        archetype: Archetype dict
        run_id: Unique identifier for this simulation run
        output_path: Where to save chart
        title: Custom title
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        if run_id:
            output_path = Config.CHART_DIR / f"default_timing_{archetype['id']}_{run_id}.png"
        else:
            output_path = Config.CHART_DIR / f"default_timing_{archetype['id']}.png"
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    if title is None:
        title = f"{archetype['name']} - Default Timing Analysis"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if result.time_to_default_percentiles:
        percentiles = result.time_to_default_percentiles
        perc_names = list(percentiles.keys())
        perc_values = [percentiles[k] for k in perc_names]
        
        ax1.barh(perc_names, perc_values, color='coral', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Month of Default', fontsize=11)
        ax1.set_ylabel('Percentile', fontsize=11)
        ax1.set_title('When Defaults Occur (Among Defaulting Paths)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        for i, (label, value) in enumerate(zip(perc_names, perc_values)):
            ax1.text(value, i, f'  Month {int(round(value))}', va='center', fontweight='bold')
        
        default_prob = result.p_default
        ax1.text(0.02, 0.98, f'Overall P(default): {default_prob:.1%}',
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No Defaults Detected\n(P(default) ~= 0%)',
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=14, fontweight='bold', color='green')
        ax1.set_title('When Defaults Occur (Among Defaulting Paths)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_risk_summary_card(
    result,
    archetype: dict,
    loan_config,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate a single-page risk summary card with key metrics.
    
    Args:
        result: SimulationResult object
        archetype: Archetype dict
        loan_config: LoanConfig object
        output_path: Where to save chart
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        output_path = Config.CHART_DIR / f"risk_summary_{archetype['id']}.png"
    
    fig = plt.figure(figsize=(12, 10))
    
    fig.suptitle(f"Risk Assessment Summary: {archetype['name']}", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.axis('off')
    
    metrics_text = [
        f"Loan: ${loan_config.amount:,.0f} @ {loan_config.annual_rate:.1%} for {loan_config.term_months} months",
        f"",
        f"RISK METRICS:",
        f"  Default Probability: {result.p_default:.2%}",
        f"  Expected Loss: ${result.expected_loss:,.2f}",
        f"  CVaR (95%): ${result.cvar_95:,.2f}",
        f"  Risk Tier: {result.recommended_loan.risk_tier.value.upper()}",
        f"",
        f"ANALYZED STRUCTURE:",
        f"  Amount: ${result.recommended_loan.optimal_amount:,.0f}",
        f"  Term: {result.recommended_loan.optimal_term_months} months",
        f"  Rate: {result.recommended_loan.optimal_rate:.1%}",
    ]
    
    ax_main.text(0.05, 0.95, '\n'.join(metrics_text),
                transform=ax_main.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax1 = fig.add_subplot(gs[1, 0])
    n_paths_to_show = min(100, result.raw_paths.shape[0])
    for i in range(n_paths_to_show):
        ax1.plot(result.raw_paths[i, :], alpha=0.1, color='steelblue', linewidth=0.5)
    ax1.plot(result.median_income_by_month, 'r-', linewidth=2, label='Median')
    ax1.set_title('Income Paths', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Income ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2 = fig.add_subplot(gs[1, 1])
    month0_income = result.raw_paths[:, 0]
    ax2.hist(month0_income, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=np.median(month0_income), color='red', linewidth=2, label='Median')
    ax2.set_title('Income Distribution (Month 0)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Income ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = fig.add_subplot(gs[2, :])
    if result.time_to_default_percentiles:
        perc_labels = list(result.time_to_default_percentiles.keys())
        perc_values = [result.time_to_default_percentiles[k] for k in perc_labels]
        
        ax3.plot(perc_labels, perc_values, marker='o', linewidth=2.5,
                markersize=8, color='darkred', label='Time to Default')
        ax3.fill_between(range(len(perc_labels)), 0, perc_values,
                         alpha=0.3, color='red')
        ax3.set_xlabel('Percentile', fontsize=11)
        ax3.set_ylabel('Month', fontsize=11)
        ax3.set_title('Default Timing Curve', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Defaults in Simulation',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=16, fontweight='bold', color='green')
        ax3.set_title('Default Timing Curve', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_risk_surface_2d(
    results_grid: List[List[Tuple[float, float, float]]],
    loan_amounts: List[float],
    loan_terms: List[int],
    output_path: Optional[Path] = None,
    title: str = "Risk Surface: Loan Amount vs Term"
) -> Path:
    """
    Plot 2D heatmap of P(default) across loan amounts and terms.
    
    Args:
        results_grid: Grid of (amount, term, p_default) tuples
        loan_amounts: List of loan amounts tested
        loan_terms: List of loan terms tested
        output_path: Where to save chart
        title: Chart title
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        output_path = Config.CHART_DIR / "risk_surface_2d.png"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    risk_matrix = np.array([[r[2] for r in row] for row in results_grid])
    
    im = ax.imshow(risk_matrix * 100, cmap='RdYlGn_r', aspect='auto',
                   extent=[loan_terms[0], loan_terms[-1], 
                          loan_amounts[0], loan_amounts[-1]])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(default) %', fontsize=12)
    
    ax.set_xlabel('Loan Term (months)', fontsize=12)
    ax.set_ylabel('Loan Amount ($)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path
