// Profile types
export {
  Platform,
  MetroArea,
  PLATFORM_DISPLAY_NAMES,
  METRO_DISPLAY_NAMES,
  DEFAULT_PROFILE,
  PROFILE_CONSTRAINTS,
} from "./profile";
export type { GigWorkerProfile } from "./profile";

// Simulation types
export {
  DEFAULT_SIMULATION_CONFIG,
  SIMULATION_PATH_OPTIONS,
  HORIZON_OPTIONS,
  percentilesToChartData,
} from "./simulation";
export type {
  SimulationConfig,
  PercentileBands,
  SimulationSummary,
  SimulationResult,
  FanChartDataPoint,
} from "./simulation";

// Loan types
export { DEFAULT_LOAN_PARAMS, LOAN_TERM_OPTIONS } from "./loan";
export type { LoanParams, LoanEvaluation } from "./loan";

// Scenario types
export { PRESET_SCENARIOS, PRESET_SCENARIO_INFO } from "./scenario";
export type {
  ScenarioAdjustments,
  MacroScenario,
  PresetScenarioId,
  ScenarioInput,
  DeltaSummary,
  CompareResult,
} from "./scenario";
