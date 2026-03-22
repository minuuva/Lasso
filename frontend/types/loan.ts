/**
 * Loan parameters - matches backend LoanParams
 */
export interface LoanParams {
  amount: number; // e.g., 15000
  annualRate: number; // e.g., 0.072 for 7.2%
  termMonths: number; // 24, 36, 48, or 60
}

/**
 * Risk assessment categories
 */
export type RiskAssessment =
  | "FICO_OVERESTIMATES_RISK" // Green - borrower is safer than FICO suggests
  | "ALIGNED" // Yellow - our model agrees with FICO
  | "FICO_UNDERESTIMATES_RISK"; // Red - borrower is riskier than FICO suggests

/**
 * Loan evaluation result - matches backend LoanEvaluation
 */
export interface LoanEvaluation {
  // Loan inputs (echoed back)
  loanAmount: number;
  annualRate: number;
  termMonths: number;
  monthlyPayment: number;

  // Our model's assessment
  probMissOnePayment: number; // P(miss >= 1 payment in loan term)
  probMissThreeConsecutive: number; // P(90+ day delinquency)
  probDefault: number; // P(default)
  monthsToFirstMissP50: number; // median months until first missed payment

  // FICO comparison
  ficoEstimatedDefaultRate: number; // from standard FICO-to-default tables
  ficoScore: number;
  riskDelta: number; // our_default_rate - fico_default_rate
  riskAssessment: RiskAssessment;
}

/**
 * Default loan parameters
 */
export const DEFAULT_LOAN_PARAMS: LoanParams = {
  amount: 15000,
  annualRate: 0.072,
  termMonths: 60,
};

/**
 * Available loan term options (in months)
 */
export const LOAN_TERM_OPTIONS = [24, 36, 48, 60] as const;

/**
 * FICO to default rate mapping (client-side for display purposes)
 */
export const FICO_DEFAULT_RATES: Array<{
  min: number;
  max: number;
  rate: number;
  label: string;
}> = [
  { min: 300, max: 579, rate: 0.28, label: "Very Poor" },
  { min: 580, max: 619, rate: 0.18, label: "Poor" },
  { min: 620, max: 659, rate: 0.11, label: "Fair" },
  { min: 660, max: 699, rate: 0.06, label: "Good" },
  { min: 700, max: 739, rate: 0.03, label: "Very Good" },
  { min: 740, max: 799, rate: 0.01, label: "Excellent" },
  { min: 800, max: 850, rate: 0.005, label: "Exceptional" },
];

/**
 * Get FICO default rate for a given credit score
 */
export function getFicoDefaultRate(creditScore: number): number {
  const bracket = FICO_DEFAULT_RATES.find(
    (b) => creditScore >= b.min && creditScore <= b.max
  );
  return bracket?.rate ?? 0.28; // Default to highest risk if out of range
}

/**
 * Get FICO rating label for a given credit score
 */
export function getFicoRatingLabel(creditScore: number): string {
  const bracket = FICO_DEFAULT_RATES.find(
    (b) => creditScore >= b.min && creditScore <= b.max
  );
  return bracket?.label ?? "Unknown";
}

/**
 * Get display color class for risk assessment
 */
export function getRiskAssessmentColor(assessment: RiskAssessment): string {
  switch (assessment) {
    case "FICO_OVERESTIMATES_RISK":
      return "text-secondary"; // Green - safer than expected
    case "ALIGNED":
      return "text-warning"; // Yellow - aligned
    case "FICO_UNDERESTIMATES_RISK":
      return "text-danger"; // Red - riskier than expected
  }
}

/**
 * Get display label for risk assessment
 */
export function getRiskAssessmentLabel(assessment: RiskAssessment): string {
  switch (assessment) {
    case "FICO_OVERESTIMATES_RISK":
      return "FICO Overestimates Risk";
    case "ALIGNED":
      return "Aligned with FICO";
    case "FICO_UNDERESTIMATES_RISK":
      return "FICO Underestimates Risk";
  }
}
