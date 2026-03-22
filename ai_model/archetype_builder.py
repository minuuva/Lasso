"""
Archetype Builder - Creates custom archetypes from user financial data.

Builds personalized archetype parameters based on actual user characteristics
rather than using pre-defined personas.
"""

import sys
import os
import numpy as np
from typing import Optional, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.loaders import DataLoader
from data_pipeline.transform.calibrate_monte_carlo import calculate_income_params
from .validation import InputValidator, ValidationResult
from .config import Config


class ArchetypeBuilder:
    """Builds custom archetypes from user data."""
    
    def __init__(self):
        """Initialize archetype builder."""
        self.loader = DataLoader()
        self.validator = InputValidator()
    
    def build_custom_archetype(
        self,
        user_data: dict,
        archetype_id: str = "custom_user"
    ) -> tuple[dict, ValidationResult]:
        """
        Build custom archetype from user financial data.
        
        Args:
            user_data: User financial information
            archetype_id: ID for the custom archetype
        
        Returns:
            (archetype_dict, validation_result)
        
        Required fields in user_data:
            - platforms: list[str]
            - hours_per_week: float
            - monthly_income_estimate: float
        
        Optional fields (with smart defaults):
            - metro_area: str
            - months_as_gig_worker: int
            - liquid_savings: float
            - monthly_fixed_expenses: float
            - existing_debt_obligations: float
            - skill_growth_rate: float
            - platform_add_rate: float
            - has_vehicle: bool
            - has_dependents: bool
        """
        validation = self.validator.validate_user_data(user_data)
        
        if not validation.valid:
            raise ValueError(
                f"Invalid user data: {validation.missing_fields}\n"
                f"Suggestions: {validation.suggestions}"
            )
        
        user_data = self.validator.apply_defaults(
            user_data,
            {**self.validator.DEFAULT_VALUES, **validation.defaults_applied}
        )
        
        platforms = user_data["platforms"]
        hours_per_week = user_data["hours_per_week"]
        income_estimate = user_data["monthly_income_estimate"]
        metro = user_data.get("metro_area", "national")
        
        income_params = calculate_income_params(
            platforms=platforms,
            hours_per_week=hours_per_week,
            metro=metro
        )
        
        base_mu = income_params["mu"]
        base_sigma = income_params["sigma"]
        
        if income_estimate and abs(income_estimate - base_mu) > base_mu * 0.3:
            adjustment_factor = income_estimate / base_mu
            base_mu = income_estimate
            base_sigma = base_sigma * adjustment_factor
        
        cv = base_sigma / base_mu if base_mu > 0 else 0.3
        
        n_platforms = len(platforms)
        experience = user_data.get("months_as_gig_worker", 12)
        
        skill_level = 1.0 + np.log(1 + experience) * 0.02
        
        skill_growth_rate = user_data.get("skill_growth_rate")
        if skill_growth_rate is None:
            if n_platforms >= 3:
                skill_growth_rate = 0.06
            elif n_platforms == 2:
                skill_growth_rate = 0.04
            else:
                skill_growth_rate = 0.02
        
        platform_add_rate = user_data.get("platform_add_rate")
        if platform_add_rate is None:
            if n_platforms >= 3:
                platform_add_rate = 0.05
            elif n_platforms == 2:
                platform_add_rate = 0.08
            else:
                platform_add_rate = 0.12
        
        churn_risk = 0.10 if n_platforms >= 2 else 0.20
        
        diversification_prob = 0.30 if n_platforms >= 2 else 0.15
        
        liquid_savings = user_data.get("liquid_savings", 1500)
        emergency_fund_weeks = int((liquid_savings / (base_mu / 4.33)))
        emergency_fund_weeks = max(1, min(emergency_fund_weeks, 12))
        
        debt = user_data.get("existing_debt_obligations", 150)
        debt_to_income_ratio = debt / base_mu if base_mu > 0 else 0.3
        
        event_modifiers = self._calculate_event_modifiers(
            cv=cv,
            n_platforms=n_platforms,
            emergency_fund_weeks=emergency_fund_weeks,
            has_vehicle=user_data.get("has_vehicle", True)
        )
        
        risk_category = self._determine_risk_category(cv, emergency_fund_weeks, debt_to_income_ratio)
        
        loan_range = self._recommend_loan_range(base_mu, risk_category)
        loan_term = self._recommend_loan_term(risk_category)
        
        archetype = {
            "id": archetype_id,
            "name": f"Custom User ({archetype_id})",
            "description": f"Custom archetype built from user data: {n_platforms} platforms, ${base_mu:.0f}/mo",
            "base_mu": float(base_mu),
            "base_sigma": float(base_sigma),
            "coefficient_of_variation": float(cv),
            "platforms": platforms,
            "hours_per_week": float(hours_per_week),
            "metro": metro,
            "experience_months": int(experience),
            "skill_level": float(skill_level),
            "skill_growth_rate": float(skill_growth_rate),
            "diversification_probability": float(diversification_prob),
            "platform_add_rate": float(platform_add_rate),
            "churn_risk": float(churn_risk),
            "event_modifiers": event_modifiers,
            "emergency_fund_weeks": int(emergency_fund_weeks),
            "debt_to_income_ratio": float(debt_to_income_ratio),
            "default_risk_category": risk_category,
            "recommended_loan_amount_range": loan_range,
            "recommended_loan_term_months": loan_term
        }
        
        return archetype, validation
    
    def _calculate_event_modifiers(
        self,
        cv: float,
        n_platforms: int,
        emergency_fund_weeks: int,
        has_vehicle: bool
    ) -> dict:
        """Calculate event probability modifiers based on user risk profile."""
        
        vehicle_mod = 1.2 if has_vehicle else 0.3
        
        health_mod = 1.0
        
        platform_deactivation_mod = 1.3 if n_platforms == 1 else 0.8
        
        housing_mod = 1.2 if emergency_fund_weeks < 3 else 0.9
        
        positive_mod = 1.3 if n_platforms >= 3 else 0.9
        
        return {
            "vehicle_repair_probability": float(vehicle_mod),
            "health_issue_probability": float(health_mod),
            "platform_deactivation_probability": float(platform_deactivation_mod),
            "housing_instability_probability": float(housing_mod),
            "positive_event_probability": float(positive_mod)
        }
    
    def _determine_risk_category(
        self,
        cv: float,
        emergency_fund_weeks: int,
        debt_to_income_ratio: float
    ) -> str:
        """Determine risk category based on key metrics."""
        
        risk_score = 0
        
        if cv > 0.35:
            risk_score += 2
        elif cv > 0.25:
            risk_score += 1
        
        if emergency_fund_weeks < 3:
            risk_score += 2
        elif emergency_fund_weeks < 5:
            risk_score += 1
        
        if debt_to_income_ratio > 0.40:
            risk_score += 2
        elif debt_to_income_ratio > 0.30:
            risk_score += 1
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _recommend_loan_range(self, base_mu: float, risk_category: str) -> list[int]:
        """Recommend loan amount range based on income and risk."""
        
        if risk_category == "high":
            return [int(base_mu * 1.0), int(base_mu * 3.0)]
        elif risk_category == "medium":
            return [int(base_mu * 2.0), int(base_mu * 6.0)]
        else:
            return [int(base_mu * 3.0), int(base_mu * 8.0)]
    
    def _recommend_loan_term(self, risk_category: str) -> int:
        """Recommend loan term based on risk category."""
        if risk_category == "high":
            return 12
        elif risk_category == "medium":
            return 24
        else:
            return 36
    
    def load_or_build_archetype(
        self,
        archetype_id: Optional[str] = None,
        user_data: Optional[dict] = None
    ) -> tuple[dict, bool]:
        """
        Load pre-defined archetype or build custom one.
        
        Args:
            archetype_id: Pre-defined archetype ID (if any)
            user_data: User data for custom archetype (if any)
        
        Returns:
            (archetype_dict, is_custom)
            - archetype_dict: Complete archetype configuration
            - is_custom: True if custom-built, False if pre-defined
        
        Raises:
            ValueError: If neither archetype_id nor user_data provided
        """
        if archetype_id and archetype_id not in ["custom", "custom_user"]:
            try:
                archetype = self.loader.load_archetype(archetype_id)
                return archetype, False
            except:
                print(f"Warning: Archetype '{archetype_id}' not found, building custom")
        
        if user_data:
            archetype, validation = self.build_custom_archetype(user_data)
            if validation.warnings:
                print(f"Validation warnings: {validation.warnings}")
            return archetype, True
        
        raise ValueError("Must provide either archetype_id or user_data")
    
    def compare_to_archetypes(self, user_data: dict) -> list[tuple[str, float]]:
        """
        Compare user data to pre-defined archetypes and suggest closest matches.
        
        Args:
            user_data: User financial data
        
        Returns:
            List of (archetype_id, similarity_score) tuples, sorted by similarity
        """
        try:
            custom_arch, _ = self.build_custom_archetype(user_data, "temp")
        except:
            return []
        
        all_archetypes = [
            "volatile_vic", "steady_sarah", "weekend_warrior",
            "sf_hustler", "rising_ryan"
        ]
        
        similarities = []
        
        for arch_id in all_archetypes:
            try:
                arch = self.loader.load_archetype(arch_id)
                score = self._calculate_similarity(custom_arch, arch)
                similarities.append((arch_id, score))
            except:
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _calculate_similarity(self, arch1: dict, arch2: dict) -> float:
        """Calculate similarity score between two archetypes."""
        
        score = 0.0
        weights = {
            "coefficient_of_variation": 0.3,
            "platforms_count": 0.2,
            "skill_growth_rate": 0.2,
            "emergency_fund_weeks": 0.15,
            "debt_to_income_ratio": 0.15
        }
        
        cv_diff = abs(arch1["coefficient_of_variation"] - arch2["coefficient_of_variation"])
        cv_similarity = max(0, 1 - cv_diff / 0.5)
        score += cv_similarity * weights["coefficient_of_variation"]
        
        n1 = len(arch1["platforms"])
        n2 = len(arch2["platforms"])
        platform_similarity = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
        score += platform_similarity * weights["platforms_count"]
        
        skill_diff = abs(arch1["skill_growth_rate"] - arch2["skill_growth_rate"])
        skill_similarity = max(0, 1 - skill_diff / 0.08)
        score += skill_similarity * weights["skill_growth_rate"]
        
        fund_diff = abs(arch1["emergency_fund_weeks"] - arch2["emergency_fund_weeks"])
        fund_similarity = max(0, 1 - fund_diff / 10)
        score += fund_similarity * weights["emergency_fund_weeks"]
        
        debt_diff = abs(arch1["debt_to_income_ratio"] - arch2["debt_to_income_ratio"])
        debt_similarity = max(0, 1 - debt_diff / 0.5)
        score += debt_similarity * weights["debt_to_income_ratio"]
        
        return score


def build_archetype_from_user_data(user_data: dict) -> dict:
    """
    Convenience function to build custom archetype.
    
    Args:
        user_data: User financial data
    
    Returns:
        Custom archetype dictionary
    """
    builder = ArchetypeBuilder()
    archetype, validation = builder.build_custom_archetype(user_data)
    
    if validation.warnings:
        print(f"Warnings: {validation.warnings}")
    if validation.suggestions:
        print(f"Suggestions: {validation.suggestions}")
    
    return archetype
