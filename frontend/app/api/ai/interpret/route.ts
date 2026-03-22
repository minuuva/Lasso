import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

const SYSTEM_PROMPT = `You are a financial scenario interpreter for Lasso, a Monte Carlo risk engine for gig workers (no credit scores—only income paths, expenses, and shocks).

Your job: turn the user's request into a structured AIScenario that the engine applies ON TOP of random life/macro path sampling (deterministic stress overlay).

AIScenario JSON shape:
{
  "narrative": "Short description",
  "parameter_shifts": [
    {
      "target": "mu_base" | "sigma_base" | "lambda" | "expenses",
      "type": "multiplicative" | "additive",
      "magnitude": number,
      "start_month": number (0 .. horizon-1, default 0 if unspecified),
      "duration_months": number (1 .. horizon, use 24-60 for slow macro shocks if horizon unknown),
      "decay": "snap_back" | "linear" | "exponential"
    }
  ],
  "discrete_jumps": [
    { "month": number, "amount": number, "variance": number }
  ]
}

Rules:
- multiplicative magnitude must stay in [0.05, 3.0]. additive expenses can be any reasonable dollar delta per month.
- Every parameter_shifts item must include exactly: target, type, magnitude, start_month, duration_months, decay.
- If the user names a catalog macro scenario, align shifts roughly with its economics (names are hints; encode as shifts/jumps):
  - recession_2008, recession_2020, inflation_slowdown_2022
  - gas_spike_moderate, gas_spike_severe
  - ab5_classification, minimum_wage_increase
  - autonomous_vehicles_pilot, ai_delivery_optimization
- For custom events ("tariffs", "pandemic", "platform ban", "accident", "new regulation"), infer plausible mu_base, sigma_base, lambda, expenses, and optional discrete_jumps (e.g. medical bill, repair).
- If they say "starting month N", set start_month = N (0-based). If unknown, use 0 or a reasonable mid-horizon month.
- Set should_run_simulation: true whenever they want to model a shock, stress, recession, spike, injury, policy change, or "what if X happens".
- Set should_run_simulation: false for pure greetings or unrelated chat.

Always respond with valid JSON including "response", "scenario", and "should_run_simulation".`;

interface AIScenario {
  narrative: string;
  parameter_shifts: Array<{
    target: "mu_base" | "sigma_base" | "lambda" | "expenses";
    type: "multiplicative" | "additive";
    magnitude: number;
    start_month: number;
    duration_months: number;
    decay: "snap_back" | "linear" | "exponential";
  }>;
  discrete_jumps: Array<{
    month: number;
    amount: number;
    variance: number;
  }>;
}

interface AIResponse {
  response: string;
  scenario: AIScenario | null;
  should_run_simulation: boolean;
}

export async function POST(request: NextRequest) {
  try {
    // Check if API key is configured
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        {
          error: "Anthropic API key not configured",
          response: "AI features are not available yet. Please add your ANTHROPIC_API_KEY to .env.local to enable scenario analysis.",
          scenario: null,
          should_run_simulation: false
        },
        { status: 503 }
      );
    }

    const body = await request.json();
    const { message, context } = body;

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    const userPrompt = `User question: "${message}"

${context ? `Current worker context: ${JSON.stringify(context)}` : ""}

Interpret this as a financial scenario for a gig worker. If it's a "what if" question or stress test, generate an AIScenario. If it's a general question, just respond conversationally.

Respond with JSON in this format:
{
  "response": "Your conversational response explaining the interpretation",
  "scenario": { /* AIScenario object */ } or null if not a scenario question,
  "should_run_simulation": true/false
}`;

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: userPrompt,
        },
      ],
    });

    // Extract text content
    const textContent = response.content.find((c) => c.type === "text");
    if (!textContent || textContent.type !== "text") {
      throw new Error("No text response from AI");
    }

    // Parse the JSON response
    let aiResponse: AIResponse;
    try {
      // Try to extract JSON from the response
      const jsonMatch = textContent.text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        aiResponse = JSON.parse(jsonMatch[0]);
      } else {
        // If no JSON found, create a default response
        aiResponse = {
          response: textContent.text,
          scenario: null,
          should_run_simulation: false,
        };
      }
    } catch {
      aiResponse = {
        response: textContent.text,
        scenario: null,
        should_run_simulation: false,
      };
    }

    return NextResponse.json(aiResponse);
  } catch (error) {
    console.error("AI interpretation error:", error);
    return NextResponse.json(
      { error: "Failed to interpret scenario", details: String(error) },
      { status: 500 }
    );
  }
}
