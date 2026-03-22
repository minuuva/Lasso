import type { LoanEvaluation } from "./loan";

/**
 * Simulation configuration - matches backend SimulationConfig
 */
export interface SimulationConfig {
  numPaths: number; // 1000, 5000, or 10000
  horizonMonths: number; // 12, 24, 36, 48, or 60
  seed?: number; // optional, for reproducibility
}

/**
 * Percentile bands for fan chart visualization
 */
export interface PercentileBands {
  p10: number[]; // 10th percentile, array length = horizonMonths
  p25: number[]; // 25th percentile
  p50: number[]; // 50th percentile (median)
  p75: number[]; // 75th percentile
  p90: number[]; // 90th percentile
  mean: number[]; // mean values
}

/**
 * Summary statistics - matches backend SimulationSummary
 */
export interface SimulationSummary {
  meanMonthlyIncome: number;
  medianMonthlyIncome: number;
  incomeVolatilityCv: number; // coefficient of variation
  probNegativeCashFlowAnyMonth: number; // % of paths with at least 1 negative month
  probSavingsDepleted: number; // % of paths where savings hit $0
  medianSavingsAtEnd: number;
  worstMonthP10: number; // 10th percentile of worst single month
}

/**
 * Full simulation result - matches backend SimulationResult
 */
export interface SimulationResult {
  incomePercentiles: PercentileBands;
  cashFlowPercentiles: PercentileBands;
  savingsPercentiles: PercentileBands;
  summary: SimulationSummary;
  loanEvaluation?: LoanEvaluation; // populated if loan params provided
}

/**
 * Chart data point for Recharts fan chart
 */
export interface FanChartDataPoint {
  month: number;
  monthLabel: string; // "Month 1", "Month 2", etc.
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  mean: number;
  // For stressed scenario overlay
  stressedP10?: number;
  stressedP25?: number;
  stressedP50?: number;
  stressedP75?: number;
  stressedP90?: number;
}

/**
 * Default simulation config
 */
export const DEFAULT_SIMULATION_CONFIG: SimulationConfig = {
  numPaths: 5000,
  horizonMonths: 36,
};

/**
 * Available simulation path counts
 */
export const SIMULATION_PATH_OPTIONS = [1000, 5000, 10000] as const;

/**
 * Available time horizon options (in months)
 */
export const HORIZON_OPTIONS = [12, 24, 36, 48, 60] as const;

/**
 * Transform percentile bands to chart data points
 */
export function percentilesToChartData(
  percentiles: PercentileBands,
  stressedPercentiles?: PercentileBands
): FanChartDataPoint[] {
  return percentiles.p50.map((_, index) => ({
    month: index + 1,
    monthLabel: `Month ${index + 1}`,
    p10: percentiles.p10[index],
    p25: percentiles.p25[index],
    p50: percentiles.p50[index],
    p75: percentiles.p75[index],
    p90: percentiles.p90[index],
    mean: percentiles.mean[index],
    ...(stressedPercentiles && {
      stressedP10: stressedPercentiles.p10[index],
      stressedP25: stressedPercentiles.p25[index],
      stressedP50: stressedPercentiles.p50[index],
      stressedP75: stressedPercentiles.p75[index],
      stressedP90: stressedPercentiles.p90[index],
    }),
  }));
}
