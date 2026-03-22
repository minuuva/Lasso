"""
Input validation for AI Layer.

Validates user data, queries, and simulation parameters.
"""

from typing import Optional, Any
from dataclasses import dataclass
from .config import Config


@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    missing_fields: list[str]
    warnings: list[str]
    suggestions: list[str]
    defaults_applied: dict[str, Any]
    
    def __repr__(self):
        if self.valid:
            return f"ValidationResult(valid=True, warnings={len(self.warnings)})"
        return f"ValidationResult(valid=False, missing={self.missing_fields})"


class InputValidator:
    """Validates user inputs and simulation parameters."""
    
    REQUIRED_FIELDS = ["platforms", "hours_per_week", "monthly_income_estimate"]
    RECOMMENDED_FIELDS = [
        "liquid_savings",
        "monthly_fixed_expenses",
        "existing_debt_obligations",
    ]
    
    VALID_PLATFORMS = [
        "uber", "lyft", "doordash", "instacart", "grubhub",
        "postmates", "shipt", "favor", "gopuff"
    ]
    
    VALID_METRO_AREAS = [
        "national", "san_francisco", "new_york", "atlanta", "dallas", "rural"
    ]
    
    DEFAULT_VALUES = {
        "metro_area": "national",
        "months_as_gig_worker": 12,
        "has_vehicle": True,
        "has_dependents": False,
        "liquid_savings": 1500,
        "monthly_fixed_expenses": 1200,
        "existing_debt_obligations": 150,
        "emergency_fund_weeks": 3,
        "skill_growth_rate": 0.04,
        "platform_add_rate": 0.08,
        "churn_risk": 0.15
    }
    
    @classmethod
    def validate_user_data(cls, user_data: dict) -> ValidationResult:
        """
        Validate user financial data.
        
        Args:
            user_data: Dictionary with user financial information
        
        Returns:
            ValidationResult with validation status and suggestions
        """
        missing_fields = []
        warnings = []
        suggestions = []
        defaults_applied = {}
        
        for field in cls.REQUIRED_FIELDS:
            if field not in user_data or user_data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            return ValidationResult(
                valid=False,
                missing_fields=missing_fields,
                warnings=[],
                suggestions=[f"Please provide: {', '.join(missing_fields)}"],
                defaults_applied={}
            )
        
        for field in cls.RECOMMENDED_FIELDS:
            if field not in user_data or user_data[field] is None:
                warnings.append(f"Missing recommended field: {field}")
                defaults_applied[field] = cls.DEFAULT_VALUES.get(field)
                suggestions.append(
                    f"Using default {field}={cls.DEFAULT_VALUES.get(field)} (consider providing actual value)"
                )
        
        if "platforms" in user_data:
            invalid_platforms = [
                p for p in user_data["platforms"]
                if p.lower() not in cls.VALID_PLATFORMS
            ]
            if invalid_platforms:
                warnings.append(f"Unknown platforms: {invalid_platforms}")
                suggestions.append(f"Valid platforms: {cls.VALID_PLATFORMS}")
        
        if "metro_area" in user_data:
            if user_data["metro_area"].lower() not in cls.VALID_METRO_AREAS:
                warnings.append(f"Unknown metro area: {user_data['metro_area']}")
                suggestions.append(f"Valid metro areas: {cls.VALID_METRO_AREAS}")
                defaults_applied["metro_area"] = "national"
        
        return ValidationResult(
            valid=True,
            missing_fields=[],
            warnings=warnings,
            suggestions=suggestions,
            defaults_applied=defaults_applied
        )
    
    @classmethod
    def validate_query(cls, query: str) -> ValidationResult:
        """
        Validate natural language query.
        
        Args:
            query: User's natural language query
        
        Returns:
            ValidationResult
        """
        warnings = []
        suggestions = []
        
        if not query or len(query.strip()) == 0:
            return ValidationResult(
                valid=False,
                missing_fields=["query"],
                warnings=[],
                suggestions=["Please provide a non-empty query"],
                defaults_applied={}
            )
        
        if len(query) > Config.MAX_QUERY_LENGTH:
            warnings.append(f"Query exceeds {Config.MAX_QUERY_LENGTH} characters")
            suggestions.append("Consider shortening your query for better results")
        
        return ValidationResult(
            valid=True,
            missing_fields=[],
            warnings=warnings,
            suggestions=suggestions,
            defaults_applied={}
        )
    
    @classmethod
    def validate_simulation_params(cls, params: dict) -> ValidationResult:
        """
        Validate extracted simulation parameters.
        
        Args:
            params: Extracted simulation parameters
        
        Returns:
            ValidationResult
        """
        warnings = []
        suggestions = []
        defaults_applied = {}
        
        if "time_horizon_months" in params:
            horizon = params["time_horizon_months"]
            if horizon < Config.MIN_TIME_HORIZON_MONTHS:
                warnings.append(f"Time horizon {horizon} too short")
                defaults_applied["time_horizon_months"] = Config.MIN_TIME_HORIZON_MONTHS
            elif horizon > Config.MAX_TIME_HORIZON_MONTHS:
                warnings.append(f"Time horizon {horizon} too long")
                defaults_applied["time_horizon_months"] = Config.MAX_TIME_HORIZON_MONTHS
        
        if "n_paths" in params:
            n_paths = params["n_paths"]
            if n_paths < Config.MIN_N_PATHS:
                warnings.append(f"n_paths {n_paths} too low for accurate results")
                defaults_applied["n_paths"] = Config.MIN_N_PATHS
            elif n_paths > Config.MAX_N_PATHS:
                warnings.append(f"n_paths {n_paths} too high (will be slow)")
                defaults_applied["n_paths"] = Config.MAX_N_PATHS
        
        return ValidationResult(
            valid=True,
            missing_fields=[],
            warnings=warnings,
            suggestions=suggestions,
            defaults_applied=defaults_applied
        )
    
    @classmethod
    def apply_defaults(cls, user_data: dict, defaults: dict) -> dict:
        """
        Apply default values to user data.
        
        Args:
            user_data: User-provided data
            defaults: Default values to apply
        
        Returns:
            User data with defaults applied
        """
        result = user_data.copy()
        for key, value in defaults.items():
            if key not in result or result[key] is None:
                result[key] = value
        return result
