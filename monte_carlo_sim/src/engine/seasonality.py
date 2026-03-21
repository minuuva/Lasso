"""
Seasonal (mu, sigma) multipliers by gig type and calendar month (0 = January … 11 = December).

Loads seasonality data from data_pipeline/data/seasonality.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.types import GigType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline.loaders import DataLoader
except ImportError:
    raise ImportError(
        "data_pipeline not found. Ensure the data_pipeline directory exists "
        "at the repository root and contains seasonality.json."
    )

# GigType -> list of 12 (mu_multiplier, sigma_multiplier)
_SEASONALITY_TABLE: dict[GigType, list[tuple[float, float]]] = {}
_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _load_from_data_pipeline() -> dict[GigType, list[tuple[float, float]]]:
    """
    Load seasonality multipliers from data_pipeline/data/seasonality.json.
    
    Returns table mapping GigType to 12 monthly (mu_mult, sigma_mult) tuples.
    Pipeline provides mu multipliers; sigma defaults to 1.0 (could be added to JSON later).
    """
    loader = DataLoader()
    
    table: dict[GigType, list[tuple[float, float]]] = {}
    
    gig_type_map = {
        GigType.DELIVERY: "delivery",
        GigType.RIDESHARE: "rideshare",
        GigType.FREELANCE: "general_gig",
    }
    
    for gig_type, json_key in gig_type_map.items():
        monthly_mults = []
        season_data = loader.get_seasonality(json_key)
        
        for month_name in _MONTH_NAMES:
            mu_mult = season_data.get(month_name, 1.0)
            sigma_mult = 1.0
            monthly_mults.append((float(mu_mult), float(sigma_mult)))
        
        table[gig_type] = monthly_mults
    
    delivery_mults = table[GigType.DELIVERY]
    freelance_mults = table[GigType.FREELANCE]
    mixed_mults = [
        ((delivery_mults[i][0] + freelance_mults[i][0]) / 2.0,
         (delivery_mults[i][1] + freelance_mults[i][1]) / 2.0)
        for i in range(12)
    ]
    table[GigType.MIXED] = mixed_mults
    
    return table


def _ensure_table() -> None:
    """Load seasonality table from data pipeline on first access."""
    if _SEASONALITY_TABLE:
        return
    loaded = _load_from_data_pipeline()
    _SEASONALITY_TABLE.update(loaded)


def get_multipliers(gig_type: GigType, month_index: int) -> tuple[float, float]:
    """
    Return (mu_multiplier, sigma_multiplier) for a calendar month.

    Parameters
    ----------
    gig_type:
        Stream gig type used for lookup.
    month_index:
        0 = January through 11 = December.
    """
    if month_index < 0 or month_index > 11:
        raise ValueError("month_index must be in [0, 11]")
    _ensure_table()
    return _SEASONALITY_TABLE[gig_type][month_index]
