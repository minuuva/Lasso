import type { SimulationResult } from "./simulation";

/**
 * Scenario adjustments - matches backend schema from AI agent output
 */
export interface ScenarioAdjustments {
  demandMultiplier?: number; // 1.0 = normal, 0.7 = 30% demand drop
  gasPriceOverride?: number; // dollars per gallon
  unemploymentRate?: number;
  hoursReductionPct?: number; // 0.0 to 1.0
  extraExpenseMonthly?: number;
  tipMultiplier?: number; // 1.0 = normal
}

/**
 * Macro scenario - full scenario definition
 */
export interface MacroScenario {
  scenarioName: string;
  description: string;
  durationMonths: number;
  startMonth: number; // 0-indexed from simulation start
  adjustments: ScenarioAdjustments;
}

/**
 * Preset scenario IDs
 */
export type PresetScenarioId =
  | "recession_2008"
  | "recession_2020"
  | "gas_spike"
  | "platform_pay_cut"
  | "injury";

/**
 * Preset scenarios for quick stress testing
 */
export const PRESET_SCENARIOS: Record<PresetScenarioId, MacroScenario> = {
  recession_2008: {
    scenarioName: "2008 Financial Crisis",
    description: "Severe economic downturn with reduced consumer spending",
    durationMonths: 18,
    startMonth: 0,
    adjustments: {
      demandMultiplier: 0.7,
      tipMultiplier: 0.75,
      hoursReductionPct: 0.2,
    },
  },

  recession_2020: {
    scenarioName: "COVID-19 Pandemic",
    description: "Initial pandemic shock followed by partial recovery",
    durationMonths: 6,
    startMonth: 0,
    adjustments: {
      demandMultiplier: 0.5,
      hoursReductionPct: 0.4,
      extraExpenseMonthly: 200, // PPE, sanitizer, etc.
    },
  },

  gas_spike: {
    scenarioName: "Gas Price Surge",
    description: "Gas prices spike to $6/gallon for 6 months",
    durationMonths: 6,
    startMonth: 0,
    adjustments: {
      gasPriceOverride: 6.0,
      demandMultiplier: 0.95, // Slight demand reduction
    },
  },

  platform_pay_cut: {
    scenarioName: "Platform Pay Cut",
    description: "Platform reduces driver/courier pay by 15%",
    durationMonths: 36, // Permanent
    startMonth: 0,
    adjustments: {
      demandMultiplier: 0.85,
    },
  },

  injury: {
    scenarioName: "Personal Injury",
    description: "Unable to work for 2 months due to injury",
    durationMonths: 2,
    startMonth: 0,
    adjustments: {
      hoursReductionPct: 1.0,
      extraExpenseMonthly: 3000, // Medical bills
    },
  },
};

/**
 * Scenario input for natural language processing
 */
export interface ScenarioInput {
  prompt: string;
}

/**
 * Delta summary showing changes between baseline and stressed
 */
export interface DeltaSummary {
  defaultProbChange: number; // stressed.probDefault - baseline.probDefault
  medianIncomeChange: number;
  savingsDepletedChange: number;
  scenarioApplied: MacroScenario;
}

/**
 * Compare simulation result (baseline vs stressed)
 */
export interface CompareResult {
  baseline: SimulationResult;
  stressed: SimulationResult;
  delta: DeltaSummary;
}

/**
 * Preset scenario display info
 */
export const PRESET_SCENARIO_INFO: Record<
  PresetScenarioId,
  { icon: string; color: string }
> = {
  recession_2008: { icon: "TrendingDown", color: "text-danger" },
  recession_2020: { icon: "AlertTriangle", color: "text-warning" },
  gas_spike: { icon: "Fuel", color: "text-warning" },
  platform_pay_cut: { icon: "DollarSign", color: "text-danger" },
  injury: { icon: "Heart", color: "text-danger" },
};
