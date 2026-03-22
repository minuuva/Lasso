/**
 * Maps UI applicant data → VarLend FastAPI /api/simulate body shapes.
 */

import type { GigWorkerProfile, LoanParams } from "@/types";
import { Platform, MetroArea } from "@/types";

const API_METROS = new Set([
  "national",
  "san_francisco",
  "new_york",
  "atlanta",
  "dallas",
  "rural",
]);

export interface AiLayerUserData {
  platforms: string[];
  hours_per_week: number;
  monthly_income_estimate: number;
  metro_area?: string;
  months_as_gig_worker?: number;
  has_vehicle?: boolean;
  has_dependents?: boolean;
  liquid_savings?: number;
  monthly_fixed_expenses?: number;
  existing_debt_obligations?: number;
}

export interface AiLayerLoanPreferences {
  amount?: number;
  term_months?: number;
  max_rate?: number;
}

export interface AiLayerChartInfo {
  type: string;
  path: string;
  description: string;
}

export interface AiLayerSimulateResponse {
  summary: string;
  quick_summary: string;
  charts: AiLayerChartInfo[];
  metrics: {
    p_default: number;
    expected_loss: number;
    cvar_95: number;
    risk_tier: string;
    approved: boolean;
  };
  trajectory_info?: Record<string, unknown>;
  archetype_info?: Record<string, unknown>;
  warnings?: string[];
  execution_time_seconds?: number;
  error?: string;
  detail?: string;
}

/** Normalize multi-select labels from extract-params to Python VALID_PLATFORMS. */
export function normalizePlatformLabel(label: string): string {
  const key = label.trim().toLowerCase().replace(/\s+/g, "");
  const map: Record<string, string> = {
    uber: "uber",
    lyft: "lyft",
    doordash: "doordash",
    grubhub: "grubhub",
    instacart: "instacart",
    ubereats: "ubereats",
  };
  return map[key] ?? key;
}

function coerceApiMetro(snake: string): string {
  const m = snake.toLowerCase();
  return API_METROS.has(m) ? m : "national";
}

export type ExtractedParamsShape = {
  platforms: string[] | null;
  hoursPerWeek: number | null;
  monthsExperience: number | null;
  metroArea: string | null;
  hasVehicle: boolean | null;
  hasDependents: boolean | null;
  liquidSavings: number | null;
  monthlyExpenses: number | null;
  existingDebt: number | null;
  loanAmount: number | null;
  loanTermMonths: number | string | null;
};

export function extractedParamsToAiPayload(
  params: ExtractedParamsShape,
  termMonths: number,
  metroSnake: string
): { userData: AiLayerUserData; loanPreferences: AiLayerLoanPreferences } {
  const platforms = (params.platforms ?? []).map(normalizePlatformLabel);
  const hours = Math.max(1, params.hoursPerWeek ?? 30);
  const incomeEstimate = Math.round(hours * 4.33 * 22);

  const userData: AiLayerUserData = {
    platforms: platforms.length > 0 ? platforms : ["doordash"],
    hours_per_week: hours,
    monthly_income_estimate: incomeEstimate,
    metro_area: coerceApiMetro(metroSnake),
    months_as_gig_worker: params.monthsExperience ?? 12,
    has_vehicle: params.hasVehicle ?? true,
    has_dependents: params.hasDependents ?? false,
    liquid_savings: params.liquidSavings ?? 0,
    monthly_fixed_expenses: params.monthlyExpenses ?? 0,
    existing_debt_obligations: params.existingDebt ?? 0,
  };

  const loanPreferences: AiLayerLoanPreferences = {
    amount: params.loanAmount ?? 5000,
    term_months: termMonths,
    max_rate: 0.2,
  };

  return { userData, loanPreferences };
}

function platformEnumToApi(p: Platform): string {
  const m: Partial<Record<Platform, string>> = {
    [Platform.UBER_RIDESHARE]: "uber",
    [Platform.LYFT]: "lyft",
    [Platform.UBER_EATS]: "ubereats",
    [Platform.DOORDASH]: "doordash",
    [Platform.INSTACART]: "instacart",
    [Platform.TASKRABBIT]: "favor",
    [Platform.MULTI_PLATFORM]: "uber",
  };
  return m[p] ?? "doordash";
}

function metroEnumToApi(m: MetroArea): string {
  const map: Record<MetroArea, string> = {
    [MetroArea.SAN_FRANCISCO]: "san_francisco",
    [MetroArea.NEW_YORK]: "new_york",
    [MetroArea.LOS_ANGELES]: "national",
    [MetroArea.CHICAGO]: "national",
    [MetroArea.ATLANTA]: "atlanta",
    [MetroArea.DALLAS]: "dallas",
    [MetroArea.WASHINGTON_DC]: "national",
    [MetroArea.RICHMOND]: "national",
  };
  return coerceApiMetro(map[m] ?? "national");
}

export function gigProfileToAiPayload(
  profile: GigWorkerProfile,
  loan: LoanParams
): { userData: AiLayerUserData; loanPreferences: AiLayerLoanPreferences } {
  const hours = Math.max(1, profile.hoursPerWeek);
  const incomeEstimate = Math.round(hours * 4.33 * 22);

  const userData: AiLayerUserData = {
    platforms:
      profile.platform === Platform.MULTI_PLATFORM
        ? ["uber", "doordash"]
        : [platformEnumToApi(profile.platform)],
    hours_per_week: hours,
    monthly_income_estimate: incomeEstimate,
    metro_area: metroEnumToApi(profile.metro),
    months_as_gig_worker: profile.monthsExperience,
    has_vehicle: true,
    has_dependents: profile.dependents > 0,
    liquid_savings: profile.currentSavings,
    monthly_fixed_expenses: profile.monthlyRent + profile.monthlyFixedExpenses,
    existing_debt_obligations: 0,
  };

  return {
    userData,
    loanPreferences: {
      amount: loan.amount,
      term_months: loan.termMonths,
      max_rate: loan.annualRate + 0.05,
    },
  };
}

export function chartPathToProxyUrl(chartPath: string): string {
  const trimmed = chartPath.replace(/^\/charts\//, "").replace(/^\//, "");
  const segments = trimmed.split("/").map((s) => encodeURIComponent(s));
  return `/api/ai/charts/${segments.join("/")}`;
}

export const BASELINE_SIMULATION_QUERY =
  "Run a full Monte Carlo risk assessment for this gig worker loan applicant using their supplied income, expenses, savings, and loan request.";
