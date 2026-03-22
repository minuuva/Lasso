"""
Income Path Plotter - Visualizes Monte Carlo income trajectories.

Generates charts showing income paths with percentile bands.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Optional


def plot_income_paths(
    result,
    archetype: dict,
    run_id: str = None,
    output_path: Optional[Path] = None,
    n_paths_to_show: int = 200,
    title: Optional[str] = None
) -> Path:
    """
    Plot income paths with percentile bands.
    
    Args:
        result: SimulationResult object
        archetype: Archetype dict with base parameters
        run_id: Unique identifier for this simulation run
        output_path: Where to save chart (auto-generated if None)
        n_paths_to_show: Number of individual paths to display
        title: Custom chart title
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        if run_id:
            output_path = Config.CHART_DIR / f"income_paths_{archetype['id']}_{run_id}.png"
        else:
            output_path = Config.CHART_DIR / f"income_paths_{archetype['id']}.png"
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_paths = result.raw_paths.shape[0]
    n_months = result.raw_paths.shape[1]
    
    for i in range(min(n_paths_to_show, n_paths)):
        ax.plot(result.raw_paths[i, :], alpha=0.05, color='steelblue', linewidth=0.5)
    
    ax.plot(result.median_income_by_month, 'r-', linewidth=3, label='Median (P50)', zorder=10)
    ax.plot(result.p10_income_by_month, 'orange', linewidth=2.5, 
            label='P10 (worst 10%)', linestyle='--', zorder=10)
    ax.plot(result.p90_income_by_month, 'green', linewidth=2.5,
            label='P90 (best 10%)', linestyle='--', zorder=10)
    
    # Use actual mean from simulation (accounting for life events) instead of theoretical base_mu
    actual_mean = float(np.mean(result.raw_paths[:, 0]))
    ax.axhline(y=actual_mean, color='black', linestyle=':', linewidth=2,
               label=f'Base Income: ${actual_mean:.0f}', alpha=0.7)
    
    if title is None:
        title = f"{archetype['name']} - Income Paths ({n_paths} simulations, {n_months} months)"
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Monthly Income ($)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Set x-axis to use integer ticks only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, n_months - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_income_distribution(
    result,
    archetype: dict,
    month: int = 0,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> Path:
    """
    Plot income distribution histogram for a specific month.
    
    Args:
        result: SimulationResult object
        archetype: Archetype dict
        month: Which month to visualize (0-indexed)
        output_path: Where to save chart
        title: Custom chart title
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        output_path = Config.CHART_DIR / f"income_dist_month{month}_{archetype['id']}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    income_data = result.raw_paths[:, month]
    
    ax.hist(income_data, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
    
    mean_val = np.mean(income_data)
    median_val = np.median(income_data)
    p10_val = np.percentile(income_data, 10)
    
    ax.axvline(x=mean_val, color='red', linewidth=2.5,
               label=f'Mean: ${mean_val:,.0f}', linestyle='-')
    ax.axvline(x=median_val, color='orange', linewidth=2.5,
               label=f'Median: ${median_val:,.0f}', linestyle='--')
    ax.axvline(x=p10_val, color='darkred', linewidth=2,
               label=f'P10: ${p10_val:,.0f}', linestyle=':')
    
    if title is None:
        title = f"{archetype['name']} - Income Distribution at Month {month}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Monthly Income ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_net_cash_flow(
    result,
    archetype: dict,
    loan_config,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> Path:
    """
    Plot net cash flow after expenses and loan payment.
    
    Args:
        result: SimulationResult object
        archetype: Archetype dict
        loan_config: LoanConfig object
        output_path: Where to save chart
        title: Custom chart title
    
    Returns:
        Path to saved chart
    """
    if output_path is None:
        from ..config import Config
        output_path = Config.CHART_DIR / f"net_cashflow_{archetype['id']}.png"
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    base_mu = archetype["base_mu"]
    debt_ratio = archetype["debt_to_income_ratio"]
    total_expenses = base_mu * 0.25 + base_mu * debt_ratio
    
    monthly_rate = loan_config.annual_rate / 12
    n_payments = loan_config.term_months
    monthly_payment = (loan_config.amount * monthly_rate * (1 + monthly_rate) ** n_payments) / \
                     ((1 + monthly_rate) ** n_payments - 1)
    
    total_obligations = total_expenses + monthly_payment
    
    net_cf = result.raw_paths - total_obligations
    
    n_paths_to_show = min(200, result.raw_paths.shape[0])
    for i in range(n_paths_to_show):
        ax.plot(net_cf[i, :], alpha=0.05, color='steelblue', linewidth=0.5)
    
    median_net = np.median(net_cf, axis=0)
    ax.plot(median_net, 'r-', linewidth=3, label='Median Net Cash Flow', zorder=10)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Break-even')
    ax.fill_between(range(result.raw_paths.shape[1]), 0, -monthly_payment,
                    alpha=0.2, color='red', label='Danger Zone')
    
    if title is None:
        title = f"{archetype['name']} - Net Cash Flow (Income - Expenses - Loan Payment)"
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Net Cash Flow ($)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-axis to use integer ticks only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, result.raw_paths.shape[1] - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path
