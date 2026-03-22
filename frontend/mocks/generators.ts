import type {
  GigWorkerProfile,
  SimulationResult,
  PercentileBands,
  SimulationSummary,
  LoanEvaluation,
  LoanParams,
  RiskAssessment,
} from "@/types";
import {
  PLATFORM_BASE_RATES,
  METRO_MULTIPLIERS,
  getSeasonalMultiplier,
  ARCHETYPES,
  INCOME_VOLATILITY,
  LIFE_EVENTS,
  type Archetype,
} from "@/lib/constants";
import { getFicoDefaultRate } from "@/types/loan";

// Seeded random number generator for consistent SSR/client values
function createSeededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Global seeded random instance - consistent across SSR and client
let globalRandom = createSeededRandom(42);

// Reset the random generator with a new seed
function resetRandom(seed: number) {
  globalRandom = createSeededRandom(seed);
}

// Get a seeded random value
function seededRandom(): number {
  return globalRandom();
}

/**
 * Calculate base monthly income from profile
 */
function calculateBaseIncome(profile: GigWorkerProfile): number {
  const platformRates = PLATFORM_BASE_RATES[profile.platform];
  const metroMult = METRO_MULTIPLIERS[profile.metro];

  if (!platformRates || !metroMult) {
    return 3000; // Fallback
  }

  const hourlyRate = platformRates.mean * metroMult.earnings;
  const monthlyHours = profile.hoursPerWeek * 4.33;

  return Math.round(hourlyRate * monthlyHours);
}

/**
 * Get platform type for seasonality
 */
function getPlatformType(platform: string): "delivery" | "rideshare" | "general_gig" {
  const platformConfig = PLATFORM_BASE_RATES[platform];
  return platformConfig?.type ?? "general_gig";
}

/**
 * Generate realistic percentile bands for fan chart
 * Creates the "spreading fan" effect over time
 * Uses platform-specific seasonality from real data
 */
export function generateMockPercentiles(
  horizonMonths: number,
  baseValue: number,
  options: {
    growthRate?: number;
    volatilityGrowth?: number;
    isIncome?: boolean;
    platformType?: "delivery" | "rideshare" | "general_gig";
    coefficientOfVariation?: number;
  } = {}
): PercentileBands {
  const {
    growthRate = 0.003,
    volatilityGrowth = 0.02,
    isIncome = true,
    platformType = "general_gig",
    coefficientOfVariation = INCOME_VOLATILITY.medianCv,
  } = options;

  const p10: number[] = [];
  const p25: number[] = [];
  const p50: number[] = [];
  const p75: number[] = [];
  const p90: number[] = [];
  const mean: number[] = [];

  for (let month = 0; month < horizonMonths; month++) {
    // Base grows slightly over time (skill improvement)
    const monthlyBase = baseValue * (1 + growthRate * month);

    // Volatility spreads out over time (fan effect)
    // Use actual CV from calibration data
    const baseSpread = monthlyBase * coefficientOfVariation;
    const timeSpread = baseSpread * (1 + volatilityGrowth * Math.sqrt(month));

    // Apply platform-specific seasonal variation for income
    const seasonalMultiplier = isIncome
      ? getSeasonalMultiplier(month % 12, platformType)
      : 1;
    const adjusted = monthlyBase * seasonalMultiplier;

    // Generate percentiles with realistic spreads
    // Add some random variation to make it look more realistic
    const noise = () => (seededRandom() - 0.5) * timeSpread * 0.1;

    p10.push(Math.max(0, Math.round(adjusted - timeSpread * 1.5 + noise())));
    p25.push(Math.max(0, Math.round(adjusted - timeSpread * 0.8 + noise())));
    p50.push(Math.max(0, Math.round(adjusted + noise())));
    p75.push(Math.round(adjusted + timeSpread * 0.8 + noise()));
    p90.push(Math.round(adjusted + timeSpread * 1.5 + noise()));
    mean.push(Math.round(adjusted * 1.02 + noise())); // Mean slightly above median
  }

  return { p10, p25, p50, p75, p90, mean };
}

/**
 * Convert an archetype to a GigWorkerProfile
 */
export function archetypeToProfile(archetype: Archetype): GigWorkerProfile {
  // Map archetype platform to our Platform enum
  const platformMap: Record<string, string> = {
    uber: "uber_rideshare",
    doordash: "doordash",
    instacart: "instacart",
    lyft: "lyft",
  };

  const primaryPlatform = archetype.platforms[0];
  const platform =
    archetype.platforms.length > 1
      ? "multi_platform"
      : platformMap[primaryPlatform] ?? "multi_platform";

  // Map metro to our MetroArea enum
  const metroMap: Record<string, string> = {
    san_francisco: "san_francisco",
    new_york: "new_york",
    los_angeles: "los_angeles",
    chicago: "chicago",
    washington_dc: "washington_dc",
    richmond: "richmond",
    atlanta: "washington_dc", // Closest available
    dallas: "richmond", // Similar cost of living
    national: "washington_dc", // Default to DC
  };

  const metro = metroMap[archetype.metro] ?? "washington_dc";

  // Calculate reasonable rent based on income and DTI
  const estimatedMonthlyIncome = archetype.baseMu;
  const monthlyRent = Math.round(estimatedMonthlyIncome * 0.35); // 35% of income
  const monthlyFixedExpenses = Math.round(estimatedMonthlyIncome * 0.15);
  const currentSavings = Math.round(
    archetype.emergencyFundWeeks * (estimatedMonthlyIncome / 4)
  );

  // Pick credit score from middle of range
  const creditScore = Math.round(
    (archetype.creditScoreRange[0] + archetype.creditScoreRange[1]) / 2
  );

  return {
    platform: platform as GigWorkerProfile["platform"],
    metro: metro as GigWorkerProfile["metro"],
    hoursPerWeek: archetype.hoursPerWeek,
    monthsExperience: archetype.experienceMonths,
    hasSecondaryIncome: false,
    secondaryMonthlyIncome: 0,
    monthlyRent,
    monthlyFixedExpenses,
    currentSavings,
    creditScore,
    dependents: 0,
  };
}

/**
 * Get all archetypes as profiles for quick selection
 */
export function getArchetypeProfiles(): Array<{
  archetype: Archetype;
  profile: GigWorkerProfile;
}> {
  return ARCHETYPES.map((archetype) => ({
    archetype,
    profile: archetypeToProfile(archetype),
  }));
}

/**
 * Generate mock summary statistics
 * Uses real life event probabilities from data_pipeline
 */
function generateMockSummary(
  profile: GigWorkerProfile,
  baseIncome: number
): SimulationSummary {
  // Add some variability based on profile factors
  const stabilityFactor = Math.min(profile.monthsExperience / 24, 1);
  const savingsFactor = Math.min(profile.currentSavings / 5000, 1);
  const expenses = profile.monthlyRent + profile.monthlyFixedExpenses;
  const baseCashFlow = baseIncome - expenses;

  // Calculate life event impact probability
  // Using real probabilities from data_pipeline
  const lifeEventProb =
    LIFE_EVENTS.probabilities.vehicle.majorRepair +
    LIFE_EVENTS.probabilities.health.majorIllness +
    LIFE_EVENTS.probabilities.platform.deactivation;

  // Calculate negative cash flow probability based on:
  // - Base volatility (36% CV)
  // - Life event probability
  // - Savings buffer
  const baseNegativeCashFlowProb =
    INCOME_VOLATILITY.medianCv * 0.5 + lifeEventProb * 0.3;
  const adjustedNegativeCashFlowProb = Math.max(
    0.05,
    baseNegativeCashFlowProb - stabilityFactor * 0.1 - savingsFactor * 0.1
  );

  // Calculate savings depletion risk
  const monthsOfRunway = profile.currentSavings / Math.max(expenses - baseIncome * 0.5, 1);
  const savingsDepletedProb = Math.max(
    0.01,
    0.3 - monthsOfRunway * 0.02 - stabilityFactor * 0.05
  );

  // Median savings projection
  const monthlySavingsRate = baseCashFlow * 0.5; // Assume 50% savings rate of positive CF
  const projectedSavings = profile.currentSavings + monthlySavingsRate * 12;

  return {
    meanMonthlyIncome: Math.round(baseIncome * 1.02),
    medianMonthlyIncome: Math.round(baseIncome),
    incomeVolatilityCv:
      INCOME_VOLATILITY.medianCv - stabilityFactor * 0.05 + (seededRandom() - 0.5) * 0.04,
    probNegativeCashFlowAnyMonth: Math.min(
      0.9,
      adjustedNegativeCashFlowProb + (seededRandom() - 0.5) * 0.1
    ),
    probSavingsDepleted: Math.min(0.5, savingsDepletedProb + (seededRandom() - 0.5) * 0.05),
    medianSavingsAtEnd: Math.max(0, projectedSavings * (0.8 + seededRandom() * 0.4)),
    worstMonthP10: Math.round(baseIncome * (1 - INCOME_VOLATILITY.extremeSwing * 2)),
  };
}

/**
 * Calculate monthly loan payment using standard amortization formula
 */
function calculateMonthlyPayment(loanParams: LoanParams): number {
  const monthlyRate = loanParams.annualRate / 12;
  const n = loanParams.termMonths;
  const P = loanParams.amount;

  if (monthlyRate === 0) return P / n;

  return Math.round(
    (P * (monthlyRate * Math.pow(1 + monthlyRate, n))) /
      (Math.pow(1 + monthlyRate, n) - 1)
  );
}

/**
 * Generate mock loan evaluation
 */
function generateMockLoanEvaluation(
  profile: GigWorkerProfile,
  loanParams: LoanParams
): LoanEvaluation {
  const monthlyPayment = calculateMonthlyPayment(loanParams);
  const ficoDefaultRate = getFicoDefaultRate(profile.creditScore);

  // Our model's assessment (varies based on profile factors)
  const stabilityFactor = Math.min(profile.monthsExperience / 24, 1);
  const savingsFactor = Math.min(profile.currentSavings / 5000, 1);
  const incomeFactor = profile.hasSecondaryIncome ? 0.7 : 1.0;

  // Calculate our model's default rate
  const ourDefaultRate =
    ficoDefaultRate *
    (1 - stabilityFactor * 0.3) *
    (1 - savingsFactor * 0.2) *
    incomeFactor;

  const riskDelta = ourDefaultRate - ficoDefaultRate;

  // Determine risk assessment
  let riskAssessment: RiskAssessment;
  if (riskDelta < -0.02) {
    riskAssessment = "FICO_OVERESTIMATES_RISK";
  } else if (riskDelta > 0.02) {
    riskAssessment = "FICO_UNDERESTIMATES_RISK";
  } else {
    riskAssessment = "ALIGNED";
  }

  return {
    loanAmount: loanParams.amount,
    annualRate: loanParams.annualRate,
    termMonths: loanParams.termMonths,
    monthlyPayment,
    probMissOnePayment: Math.min(ourDefaultRate * 2.5, 0.8),
    probMissThreeConsecutive: Math.min(ourDefaultRate * 1.2, 0.5),
    probDefault: ourDefaultRate,
    monthsToFirstMissP50: 12 + seededRandom() * 12,
    ficoEstimatedDefaultRate: ficoDefaultRate,
    ficoScore: profile.creditScore,
    riskDelta,
    riskAssessment,
  };
}

/**
 * Generate complete mock simulation result
 * Uses real calibration data from data_pipeline
 */
export function generateMockSimulationResult(
  profile: GigWorkerProfile,
  horizonMonths: number,
  loanParams?: LoanParams
): SimulationResult {
  const baseIncome = calculateBaseIncome(profile);
  const expenses = profile.monthlyRent + profile.monthlyFixedExpenses;
  const baseCashFlow = baseIncome - expenses;

  // Get platform type for seasonality
  const platformType = getPlatformType(profile.platform);

  // Calculate CV based on experience (more experience = lower volatility)
  const experienceFactor = Math.min(profile.monthsExperience / 24, 1);
  const cv = INCOME_VOLATILITY.medianCv * (1 - experienceFactor * 0.2);

  const result: SimulationResult = {
    incomePercentiles: generateMockPercentiles(horizonMonths, baseIncome, {
      growthRate: 0.003,
      volatilityGrowth: 0.015,
      isIncome: true,
      platformType,
      coefficientOfVariation: cv,
    }),
    cashFlowPercentiles: generateMockPercentiles(horizonMonths, baseCashFlow, {
      growthRate: 0.002,
      volatilityGrowth: 0.025,
      isIncome: false,
      coefficientOfVariation: cv * 1.2, // Cash flow more volatile
    }),
    savingsPercentiles: generateMockPercentiles(
      horizonMonths,
      profile.currentSavings,
      {
        growthRate: baseCashFlow > 0 ? 0.01 : -0.02,
        volatilityGrowth: 0.03,
        isIncome: false,
        coefficientOfVariation: cv * 0.8,
      }
    ),
    summary: generateMockSummary(profile, baseIncome),
  };

  if (loanParams) {
    result.loanEvaluation = generateMockLoanEvaluation(profile, loanParams);
  }

  return result;
}

/**
 * Generate stressed simulation result
 * Applies scenario adjustments to worsen outcomes
 */
export function generateStressedSimulationResult(
  profile: GigWorkerProfile,
  horizonMonths: number,
  loanParams?: LoanParams,
  demandMultiplier: number = 0.75
): SimulationResult {
  const baseResult = generateMockSimulationResult(
    profile,
    horizonMonths,
    loanParams
  );

  // Apply stress to percentiles
  const applyStress = (bands: PercentileBands): PercentileBands => ({
    p10: bands.p10.map((v) => Math.round(v * demandMultiplier * 0.9)),
    p25: bands.p25.map((v) => Math.round(v * demandMultiplier * 0.95)),
    p50: bands.p50.map((v) => Math.round(v * demandMultiplier)),
    p75: bands.p75.map((v) => Math.round(v * demandMultiplier * 1.02)),
    p90: bands.p90.map((v) => Math.round(v * demandMultiplier * 1.05)),
    mean: bands.mean.map((v) => Math.round(v * demandMultiplier)),
  });

  // Worsen loan evaluation
  let stressedLoanEval: LoanEvaluation | undefined;
  if (baseResult.loanEvaluation) {
    const stressFactor = 1 + (1 - demandMultiplier) * 2;
    stressedLoanEval = {
      ...baseResult.loanEvaluation,
      probMissOnePayment: Math.min(
        baseResult.loanEvaluation.probMissOnePayment * stressFactor,
        0.9
      ),
      probMissThreeConsecutive: Math.min(
        baseResult.loanEvaluation.probMissThreeConsecutive * stressFactor,
        0.7
      ),
      probDefault: Math.min(
        baseResult.loanEvaluation.probDefault * stressFactor,
        0.6
      ),
    };
    stressedLoanEval.riskDelta =
      stressedLoanEval.probDefault -
      stressedLoanEval.ficoEstimatedDefaultRate;
    if (stressedLoanEval.riskDelta > 0.02) {
      stressedLoanEval.riskAssessment = "FICO_UNDERESTIMATES_RISK";
    }
  }

  return {
    incomePercentiles: applyStress(baseResult.incomePercentiles),
    cashFlowPercentiles: applyStress(baseResult.cashFlowPercentiles),
    savingsPercentiles: applyStress(baseResult.savingsPercentiles),
    summary: {
      ...baseResult.summary,
      meanMonthlyIncome: Math.round(
        baseResult.summary.meanMonthlyIncome * demandMultiplier
      ),
      medianMonthlyIncome: Math.round(
        baseResult.summary.medianMonthlyIncome * demandMultiplier
      ),
      probNegativeCashFlowAnyMonth: Math.min(
        baseResult.summary.probNegativeCashFlowAnyMonth * 1.5,
        0.9
      ),
      probSavingsDepleted: Math.min(
        baseResult.summary.probSavingsDepleted * 2,
        0.5
      ),
    },
    loanEvaluation: stressedLoanEval,
  };
}
