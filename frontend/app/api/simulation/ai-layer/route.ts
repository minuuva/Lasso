import { NextRequest, NextResponse } from "next/server";

/**
 * Proxies to Python VarLend AI Layer: POST /api/simulate
 * Set AI_MODEL_API_BASE_URL in .env.local (e.g. http://127.0.0.1:8000)
 */
export async function POST(request: NextRequest) {
  const base = process.env.AI_MODEL_API_BASE_URL?.replace(/\/$/, "");
  if (!base) {
    return NextResponse.json(
      {
        error: "AI layer not configured",
        detail:
          "Set AI_MODEL_API_BASE_URL in .env.local to your FastAPI server (e.g. http://127.0.0.1:8000), then run the ai_model API from the repo root.",
      },
      { status: 503 }
    );
  }

  try {
    const body = await request.json();

    const payload = {
      query:
        typeof body.query === "string" && body.query.trim().length >= 10
          ? body.query.trim()
          : "Run baseline Monte Carlo risk assessment for this gig worker applicant with supplied financials and loan request.",
      user_data: body.userData ?? undefined,
      loan_preferences: body.loanPreferences ?? undefined,
      structured_scenario: body.structuredScenario ?? undefined,
      use_archetype: body.useArchetype ?? undefined,
      random_seed: typeof body.randomSeed === "number" ? body.randomSeed : 42,
      generate_charts: body.generateCharts !== false,
    };

    const res = await fetch(`${base}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = (await res.json()) as Record<string, unknown>;

    if (!res.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail ?? data);
      return NextResponse.json(
        { error: "Simulation failed", detail },
        { status: res.status }
      );
    }

    return NextResponse.json(data);
  } catch (e) {
    console.error("ai-layer proxy error:", e);
    return NextResponse.json(
      { error: "Proxy error", detail: String(e) },
      { status: 500 }
    );
  }
}
