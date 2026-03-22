import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import { generateMockSimulationResult } from "@/mocks/generators";
import type { GigWorkerProfile, SimulationConfig, LoanParams } from "@/types";

// Flag to use real Python backend vs mock
const USE_PYTHON_BACKEND = process.env.USE_PYTHON_BACKEND === "true";

interface PythonSimulationInput {
  profile: {
    streams: Array<{
      platform_name: string;
      gig_type: string;
      mean_monthly_income: number;
      income_variance: number;
      tenure_months: number;
      is_primary: boolean;
    }>;
    metro_area: string;
    months_as_gig_worker: number;
    has_vehicle: boolean;
    has_dependents: boolean;
    liquid_savings: number;
    monthly_fixed_expenses: number;
    existing_debt_obligations: number;
    credit_score_range: [number, number];
    loan_request_amount: number;
    requested_term_months: number;
    acceptable_rate_range: [number, number];
  };
  scenario?: {
    narrative: string;
    parameter_shifts: Array<{
      target: string;
      type: string;
      magnitude: number;
      start_month: number;
      duration_months: number;
      decay: string;
    }>;
    discrete_jumps: Array<{
      month: number;
      amount: number;
      variance: number;
    }>;
  };
  config?: {
    n_paths: number;
    horizon_months: number;
    random_seed?: number;
  };
  loan?: {
    amount: number;
    term_months: number;
    annual_rate: number;
  };
}

function convertProfileToPython(profile: GigWorkerProfile): PythonSimulationInput["profile"] {
  // Map platform to gig type
  const platformToGigType: Record<string, string> = {
    uber_rideshare: "rideshare",
    lyft: "rideshare",
    uber_eats: "delivery",
    doordash: "delivery",
    instacart: "delivery",
    taskrabbit: "freelance",
    multi_platform: "mixed",
  };

  // Estimate income based on hours and platform
  const estimatedMonthlyIncome = profile.hoursPerWeek * 4 * 25; // ~$25/hour estimate

  return {
    streams: [{
      platform_name: profile.platform,
      gig_type: platformToGigType[profile.platform] || "delivery",
      mean_monthly_income: estimatedMonthlyIncome,
      income_variance: Math.pow(estimatedMonthlyIncome * 0.35, 2),
      tenure_months: profile.monthsExperience,
      is_primary: true,
    }],
    metro_area: profile.metro || "national",
    months_as_gig_worker: profile.monthsExperience,
    has_vehicle: true, // Assumed for most gig workers
    has_dependents: profile.dependents > 0,
    liquid_savings: profile.currentSavings,
    monthly_fixed_expenses: profile.monthlyRent + profile.monthlyFixedExpenses,
    existing_debt_obligations: 0, // Not in current profile type
    credit_score_range: [profile.creditScore - 20, profile.creditScore + 20] as [number, number],
    loan_request_amount: 5000,
    requested_term_months: 24,
    acceptable_rate_range: [0.08, 0.18] as [number, number],
  };
}

async function runPythonSimulation(input: PythonSimulationInput): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(process.cwd(), "..", "monte_carlo_sim", "api_runner.py");

    const python = spawn("python3", [pythonScript], {
      cwd: path.join(process.cwd(), "..", "monte_carlo_sim"),
    });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    python.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    python.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        reject(new Error(`Failed to parse Python output: ${stdout}`));
      }
    });

    python.on("error", (err) => {
      reject(err);
    });

    // Send input to Python
    python.stdin.write(JSON.stringify(input));
    python.stdin.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const profile: GigWorkerProfile = body.profile;
    const config: SimulationConfig = body.config;
    const loanParams: LoanParams | undefined = body.loanParams;
    const scenario = body.scenario; // AI-generated scenario

    // Try Python backend if enabled
    if (USE_PYTHON_BACKEND) {
      try {
        const pythonInput: PythonSimulationInput = {
          profile: convertProfileToPython(profile),
          config: {
            n_paths: config.numPaths || 5000,
            horizon_months: config.horizonMonths || 24,
            random_seed: 42,
          },
          loan: loanParams ? {
            amount: loanParams.amount,
            term_months: loanParams.termMonths,
            annual_rate: loanParams.annualRate,
          } : undefined,
          scenario: scenario,
        };

        const pythonResult = await runPythonSimulation(pythonInput) as Record<string, unknown>;
        return NextResponse.json({
          ...pythonResult,
          source: "python",
        });
      } catch (pythonError) {
        console.error("Python backend error, falling back to mock:", pythonError);
        // Fall through to mock
      }
    }

    // Fallback to mock simulation
    await new Promise((resolve) =>
      setTimeout(resolve, 800 + Math.random() * 400)
    );

    const result = generateMockSimulationResult(
      profile,
      config.horizonMonths,
      loanParams
    );

    return NextResponse.json({
      ...result,
      source: "mock",
    });
  } catch (error) {
    console.error("Simulation error:", error);
    return NextResponse.json(
      { error: "Simulation failed" },
      { status: 500 }
    );
  }
}
