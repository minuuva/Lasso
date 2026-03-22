/**
 * Calibration data from JPMorgan Chase Institute research, FRED, and team data pipeline
 * Sources: JPMC Institute "Weathering Volatility 2.0", Gridwise, Triplog, FRED API
 *
 * NOTE: This data is synced from data_pipeline/data/*.json
 */

// =============================================================================
// INCOME VOLATILITY PARAMETERS (from data_pipeline/data/expenses.json)
// =============================================================================
export const INCOME_VOLATILITY = {
  medianCv: 0.36, // coefficient of variation
  typicalSwing: 0.09,
  extremeSwing: 0.21,
  bufferWeeks: 6,
  spikeProbability: 2.0, // spikes 2x as likely as dips
  spikeMonths: [3, 12], // March and December
} as const;

// Legacy exports for backward compatibility
export const MEDIAN_MONTHLY_CV = INCOME_VOLATILITY.medianCv;
export const INCOME_SPIKE_PROBABILITY = 0.33;
export const INCOME_DIP_PROBABILITY = 0.17;
export const SPIKE_MAGNITUDE_MEAN = 0.3;
export const DIP_MAGNITUDE_MEAN = 0.25;

// =============================================================================
// SEASONAL MULTIPLIERS (from data_pipeline/data/seasonality.json)
// =============================================================================
export const SEASONALITY = {
  delivery: {
    jan: 1.05, feb: 0.95, mar: 1.15, apr: 0.98, may: 0.92, jun: 0.88,
    jul: 0.85, aug: 0.90, sep: 0.95, oct: 1.10, nov: 1.25, dec: 1.35,
  },
  rideshare: {
    jan: 0.95, feb: 0.95, mar: 1.10, apr: 1.05, may: 1.05, jun: 1.10,
    jul: 1.15, aug: 1.10, sep: 1.00, oct: 1.05, nov: 1.15, dec: 1.20,
  },
  general_gig: {
    jan: 1.00, feb: 0.97, mar: 1.10, apr: 1.02, may: 0.98, jun: 0.95,
    jul: 0.93, aug: 0.97, sep: 1.00, oct: 1.08, nov: 1.18, dec: 1.25,
  },
} as const;

// Tax quarter due months (reduces income by ~25%)
export const TAX_QUARTERS = {
  dueMonths: ["apr", "jun", "sep", "jan"],
  effectiveIncomeReduction: 0.25,
} as const;

// Legacy array format (index 0 = January) - uses general_gig for backward compat
export const SEASONAL_MULTIPLIERS = [
  SEASONALITY.general_gig.jan, SEASONALITY.general_gig.feb,
  SEASONALITY.general_gig.mar, SEASONALITY.general_gig.apr,
  SEASONALITY.general_gig.may, SEASONALITY.general_gig.jun,
  SEASONALITY.general_gig.jul, SEASONALITY.general_gig.aug,
  SEASONALITY.general_gig.sep, SEASONALITY.general_gig.oct,
  SEASONALITY.general_gig.nov, SEASONALITY.general_gig.dec,
];

/**
 * Get seasonal multiplier for a specific platform type
 */
export function getSeasonalMultiplier(
  monthIndex: number,
  platformType: "delivery" | "rideshare" | "general_gig" = "general_gig"
): number {
  const months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"] as const;
  const monthKey = months[monthIndex % 12];
  return SEASONALITY[platformType][monthKey];
}

// =============================================================================
// PLATFORM BASE RATES (from spec + Gridwise data)
// =============================================================================
export const PLATFORM_BASE_RATES: Record<
  string,
  { mean: number; std: number; type: "delivery" | "rideshare" | "general_gig" }
> = {
  uber_rideshare: { mean: 23.33, std: 5.0, type: "rideshare" },
  uber_eats: { mean: 18.0, std: 6.0, type: "delivery" },
  doordash: { mean: 17.5, std: 5.5, type: "delivery" },
  instacart: { mean: 19.0, std: 5.0, type: "delivery" },
  lyft: { mean: 21.0, std: 5.0, type: "rideshare" },
  taskrabbit: { mean: 28.0, std: 10.0, type: "general_gig" },
  multi_platform: { mean: 21.0, std: 6.0, type: "general_gig" },
};

// =============================================================================
// METRO COST-OF-LIVING MULTIPLIERS
// =============================================================================
export const METRO_MULTIPLIERS: Record<
  string,
  { earnings: number; gasPrice: number; rentIndex: number }
> = {
  new_york: { earnings: 1.25, gasPrice: 3.8, rentIndex: 1.5 },
  los_angeles: { earnings: 1.15, gasPrice: 4.5, rentIndex: 1.35 },
  chicago: { earnings: 1.05, gasPrice: 3.6, rentIndex: 1.1 },
  san_francisco: { earnings: 1.3, gasPrice: 4.8, rentIndex: 1.6 },
  washington_dc: { earnings: 1.15, gasPrice: 3.5, rentIndex: 1.25 },
  richmond: { earnings: 0.95, gasPrice: 3.3, rentIndex: 0.85 },
  atlanta: { earnings: 1.0, gasPrice: 3.4, rentIndex: 0.95 },
  dallas: { earnings: 1.0, gasPrice: 3.2, rentIndex: 0.9 },
  national: { earnings: 1.0, gasPrice: 3.5, rentIndex: 1.0 },
};

// =============================================================================
// EXPENSE PARAMETERS (from data_pipeline/data/expenses.json)
// =============================================================================
export const BASE_EXPENSES = {
  gasWeeklyFulltimeRange: [150, 400] as const,
  gasWeeklyParttimeRange: [50, 150] as const,
  maintenanceMonthlyRange: [50, 100] as const,
  vehicleDepreciationMonthly: 200,
  insuranceMonthly: 150,
  phoneDataMonthly: 50,
  selfEmploymentTaxRate: 0.153,
} as const;

// Legacy exports
export const GAS_GALLONS_PER_HOUR = 1.2;
export const MAINTENANCE_MONTHLY_BASE = 75.0;
export const MAINTENANCE_SHOCK_PROB = 0.15; // Updated from real data
export const MAINTENANCE_SHOCK_RANGE = { min: 500, max: 2000 };
export const SELF_EMPLOYMENT_TAX_RATE = BASE_EXPENSES.selfEmploymentTaxRate;

// =============================================================================
// LIFE EVENTS (from data_pipeline/data/expenses.json)
// =============================================================================
export const LIFE_EVENTS = {
  probabilities: {
    vehicle: {
      minorRepair: 0.4,
      majorRepair: 0.15,
      accident: 0.1,
      replacementNeeded: 0.05,
    },
    health: {
      minorIllness: 0.3,
      majorIllness: 0.1,
      chronicIssue: 0.05,
    },
    platform: {
      deactivation: 0.2,
      feeIncrease: 0.25,
      marketSaturation: 0.15,
      policyChange: 0.2,
    },
    housing: {
      rentIncrease: 0.3,
      forcedMove: 0.05,
      emergencyRepair: 0.1,
    },
    positive: {
      newPlatform: 0.25,
      skillUpgrade: 0.15,
      referralBonus: 0.2,
      sideGig: 0.1,
    },
  },
  impacts: {
    vehicleMinorRepair: [-300, -150] as const,
    vehicleMajorRepair: [-2000, -500] as const,
    vehicleAccidentDowntimeWeeks: [2, 4] as const,
    healthMinorIllnessDays: [5, 7] as const,
    healthMajorIllnessWeeks: [4, 12] as const,
    platformDeactivationWeeks: [2, 8] as const,
    platformFeeIncreasePercentage: [0.02, 0.05] as const,
    rentIncreaseMonthly: [100, 300] as const,
    newPlatformIncomeBoost: [0.15, 0.3] as const,
    skillUpgradeIncomeBoost: [0.05, 0.1] as const,
  },
} as const;

// =============================================================================
// ARCHETYPES (from data_pipeline/data/archetypes.json)
// =============================================================================
export interface Archetype {
  id: string;
  name: string;
  description: string;
  baseMu: number;
  baseSigma: number;
  coefficientOfVariation: number;
  platforms: string[];
  hoursPerWeek: number;
  metro: string;
  experienceMonths: number;
  emergencyFundWeeks: number;
  debtToIncomeRatio: number;
  defaultRiskCategory: "low" | "medium" | "high";
  recommendedLoanAmountRange: [number, number];
  recommendedLoanTermMonths: number;
}

export const ARCHETYPES: Archetype[] = [
  {
    id: "volatile_vic",
    name: "Volatile Vic",
    description: "Full-time DoorDash driver, single platform, high variance income",
    baseMu: 1642.31,
    baseSigma: 679.92,
    coefficientOfVariation: 0.414,
    platforms: ["doordash"],
    hoursPerWeek: 45,
    metro: "national",
    experienceMonths: 6,
    emergencyFundWeeks: 2,
    debtToIncomeRatio: 0.45,
    defaultRiskCategory: "high",
    recommendedLoanAmountRange: [2000, 5000],
    recommendedLoanTermMonths: 12,
  },
  {
    id: "steady_sarah",
    name: "Steady Sarah",
    description: "Multi-platform gig worker, diversified income, stable earnings",
    baseMu: 1450.08,
    baseSigma: 326.51,
    coefficientOfVariation: 0.225,
    platforms: ["uber", "doordash", "instacart"],
    hoursPerWeek: 40,
    metro: "atlanta",
    experienceMonths: 24,
    emergencyFundWeeks: 8,
    debtToIncomeRatio: 0.3,
    defaultRiskCategory: "low",
    recommendedLoanAmountRange: [5000, 10000],
    recommendedLoanTermMonths: 24,
  },
  {
    id: "weekend_warrior",
    name: "Weekend Warrior",
    description: "Part-time rideshare driver, supplemental income on weekends",
    baseMu: 580.99,
    baseSigma: 209.16,
    coefficientOfVariation: 0.36,
    platforms: ["uber"],
    hoursPerWeek: 15,
    metro: "dallas",
    experienceMonths: 12,
    emergencyFundWeeks: 10,
    debtToIncomeRatio: 0.25,
    defaultRiskCategory: "low",
    recommendedLoanAmountRange: [1000, 3000],
    recommendedLoanTermMonths: 12,
  },
  {
    id: "sf_hustler",
    name: "SF Hustler",
    description: "High-volume multi-platform driver in expensive San Francisco market",
    baseMu: 3948.41,
    baseSigma: 1080.48,
    coefficientOfVariation: 0.274,
    platforms: ["uber", "doordash"],
    hoursPerWeek: 50,
    metro: "san_francisco",
    experienceMonths: 36,
    emergencyFundWeeks: 4,
    debtToIncomeRatio: 0.4,
    defaultRiskCategory: "medium",
    recommendedLoanAmountRange: [5000, 12000],
    recommendedLoanTermMonths: 24,
  },
  {
    id: "rising_ryan",
    name: "Rising Ryan",
    description: "New gig worker on growth trajectory, rapidly building skills",
    baseMu: 803.37,
    baseSigma: 230.07,
    coefficientOfVariation: 0.286,
    platforms: ["doordash", "instacart"],
    hoursPerWeek: 35,
    metro: "national",
    experienceMonths: 3,
    emergencyFundWeeks: 3,
    debtToIncomeRatio: 0.35,
    defaultRiskCategory: "medium",
    recommendedLoanAmountRange: [2000, 6000],
    recommendedLoanTermMonths: 18,
  },
];

// =============================================================================
// MACRO SCENARIOS (from data_pipeline/data/macro_params.json)
// =============================================================================
export const MACRO_SCENARIOS = {
  recession: {
    recession_2008: {
      name: "2008 Financial Crisis",
      triggerProbability: 0.05,
      durationMonths: 18,
      unemploymentDelta: 5.0,
      platformImpacts: { rideshare: 0.7, delivery: 0.85, freelance: 0.75 },
      expenseChanges: { gasPriceMultiplier: 1.45, insuranceMultiplier: 1.1 },
    },
    recession_2020: {
      name: "COVID-19 Recession",
      triggerProbability: 0.02,
      durationMonths: 2,
      unemploymentDelta: 11.2,
      platformImpacts: { rideshare: 0.4, delivery: 1.3, freelance: 1.1 },
      expenseChanges: { gasPriceMultiplier: 0.7, insuranceMultiplier: 0.95 },
    },
    inflation_2022: {
      name: "2022 Inflation Slowdown",
      triggerProbability: 0.15,
      durationMonths: 18,
      unemploymentDelta: 0.4,
      platformImpacts: { rideshare: 0.95, delivery: 1.05, freelance: 0.9 },
      expenseChanges: { gasPriceMultiplier: 1.3, insuranceMultiplier: 1.12 },
    },
  },
  gasSpike: {
    moderate: {
      name: "Moderate Gas Price Spike",
      triggerProbability: 0.2,
      durationMonths: 4,
      gasPriceMultiplier: 1.25,
      platformImpacts: { rideshare: 0.92, delivery: 0.9 },
      expenseImpact: 300,
    },
    severe: {
      name: "Severe Gas Price Spike",
      triggerProbability: 0.08,
      durationMonths: 6,
      gasPriceMultiplier: 1.5,
      platformImpacts: { rideshare: 0.85, delivery: 0.8 },
      expenseImpact: 500,
    },
  },
  baselineProbabilities: {
    recessionAnnual: 0.1,
    gasSpikeAnnual: 0.15,
    regulatoryChangeAnnual: 0.05,
    techDisruptionAnnual: 0.05,
  },
} as const;

// =============================================================================
// TIP PERCENTAGES BY PLATFORM
// =============================================================================
export const TIP_PERCENTAGE: Record<string, number> = {
  uber_rideshare: 0.15,
  uber_eats: 0.25,
  doordash: 0.3,
  instacart: 0.2,
  lyft: 0.15,
  taskrabbit: 0.1,
  multi_platform: 0.2,
};

// =============================================================================
// HOURS WORKED DISTRIBUTION
// =============================================================================
export const HOURS_CV = 0.2;
export const MIN_HOURS_WEEK = 0;
export const MAX_HOURS_WEEK = 70;

// =============================================================================
// PORTFOLIO EVOLUTION (from data_pipeline/data/expenses.json)
// =============================================================================
export const PORTFOLIO_EVOLUTION = {
  initialPlatforms: 1.0,
  month12Platforms: 2.3,
  diversificationRate: 0.1,
  platformChurnRate: 0.05,
  hourlyEarningsGrowth: 0.1,
  efficiencyCurve: "logarithmic",
  quitProbabilityAnnual: 0.15,
} as const;

// =============================================================================
// DEFAULT PARAMETERS
// =============================================================================
export const DEFAULT_PARAMS = {
  minimumMonthlyIncome: 1500,
  emergencyFundWeeks: 6,
  debtToIncomeThreshold: 0.4,
  recoveryRate: 0.3,
} as const;

// =============================================================================
// CHART COLORS
// =============================================================================
export const CHART_COLORS = {
  fan: {
    p10p25: "rgba(0, 102, 204, 0.15)",
    p25p50: "rgba(0, 102, 204, 0.3)",
    p50p75: "rgba(0, 102, 204, 0.3)",
    p75p90: "rgba(0, 102, 204, 0.15)",
    median: "#0066CC",
    mean: "#004499",
  },
  stress: {
    p10p25: "rgba(211, 47, 47, 0.1)",
    p25p50: "rgba(211, 47, 47, 0.2)",
    p50p75: "rgba(211, 47, 47, 0.2)",
    p75p90: "rgba(211, 47, 47, 0.1)",
    median: "#D32F2F",
    mean: "#B71C1C",
  },
  risk: {
    low: "#00A878",
    medium: "#FF6B35",
    high: "#D32F2F",
  },
};
