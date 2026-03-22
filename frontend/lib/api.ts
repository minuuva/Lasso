import type {
  GigWorkerProfile,
  SimulationConfig,
  SimulationResult,
  LoanParams,
  MacroScenario,
  CompareResult,
  DeltaSummary,
} from "@/types";
import {
  generateMockSimulationResult,
  generateStressedSimulationResult,
} from "@/mocks/generators";

const API_BASE = "/api";

/**
 * Simulate network delay for realistic feel
 */
async function simulateDelay(ms: number = 800): Promise<void> {
  return new Promise((resolve) =>
    setTimeout(resolve, ms + Math.random() * 400)
  );
}

/**
 * Run basic simulation (no loan evaluation)
 */
export async function runSimulation(
  profile: GigWorkerProfile,
  config: SimulationConfig
): Promise<SimulationResult> {
  // In production, this would call the backend
  // For now, use mock data generation
  await simulateDelay();

  return generateMockSimulationResult(profile, config.horizonMonths);
}

/**
 * Run simulation with loan evaluation
 */
export async function simulateWithLoan(
  profile: GigWorkerProfile,
  config: SimulationConfig,
  loanParams: LoanParams
): Promise<SimulationResult> {
  await simulateDelay(1000);

  return generateMockSimulationResult(
    profile,
    config.horizonMonths,
    loanParams
  );
}

/**
 * Compare baseline vs stressed scenario
 */
export async function compareSimulations(
  profile: GigWorkerProfile,
  config: SimulationConfig,
  loanParams: LoanParams,
  scenario: MacroScenario
): Promise<CompareResult> {
  await simulateDelay(1200);

  const baseline = generateMockSimulationResult(
    profile,
    config.horizonMonths,
    loanParams
  );

  // Calculate demand multiplier from scenario
  const demandMultiplier = scenario.adjustments.demandMultiplier ?? 1.0;

  const stressed = generateStressedSimulationResult(
    profile,
    config.horizonMonths,
    loanParams,
    demandMultiplier
  );

  const delta: DeltaSummary = {
    defaultProbChange:
      (stressed.loanEvaluation?.probDefault ?? 0) -
      (baseline.loanEvaluation?.probDefault ?? 0),
    medianIncomeChange:
      stressed.summary.medianMonthlyIncome -
      baseline.summary.medianMonthlyIncome,
    savingsDepletedChange:
      stressed.summary.probSavingsDepleted -
      baseline.summary.probSavingsDepleted,
    scenarioApplied: scenario,
  };

  return { baseline, stressed, delta };
}

/**
 * API client that can switch between mock and real backend
 */
export const api = {
  /**
   * Run simulation - switches to real API when backend is ready
   */
  simulate: async (
    profile: GigWorkerProfile,
    config: SimulationConfig,
    loanParams?: LoanParams
  ): Promise<SimulationResult> => {
    // Check if we should use real API
    const useRealApi = process.env.NEXT_PUBLIC_USE_REAL_API === "true";

    if (useRealApi) {
      const endpoint = loanParams
        ? `${API_BASE}/simulate-with-loan`
        : `${API_BASE}/simulate`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile, config, loanParams }),
      });

      if (!response.ok) {
        throw new Error(`Simulation failed: ${response.statusText}`);
      }

      return response.json();
    }

    // Use mock data
    return loanParams
      ? simulateWithLoan(profile, config, loanParams)
      : runSimulation(profile, config);
  },

  /**
   * Compare simulations with stress scenario
   */
  compare: async (
    profile: GigWorkerProfile,
    config: SimulationConfig,
    loanParams: LoanParams,
    scenario: MacroScenario
  ): Promise<CompareResult> => {
    const useRealApi = process.env.NEXT_PUBLIC_USE_REAL_API === "true";

    if (useRealApi) {
      const response = await fetch(`${API_BASE}/simulate/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile, config, loanParams, scenario }),
      });

      if (!response.ok) {
        throw new Error(`Comparison failed: ${response.statusText}`);
      }

      return response.json();
    }

    return compareSimulations(profile, config, loanParams, scenario);
  },
};

export default api;
