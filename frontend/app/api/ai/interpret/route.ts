import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

const SYSTEM_PROMPT = `You are a financial scenario interpreter for Lasso, a Monte Carlo credit risk simulation tool for gig workers.

Your job is to interpret user questions about "what if" scenarios and convert them into structured AIScenario objects that can be fed into the simulation engine.

The AIScenario has this structure:
{
  "narrative": "Human-readable description of the scenario",
  "parameter_shifts": [
    {
      "target": "mu_base" | "sigma_base" | "lambda" | "expenses",
      "type": "multiplicative" | "additive",
      "magnitude": number (0.05 to 3.0 for multiplicative, any for additive),
      "start_month": number (0 to horizon-1),
      "duration_months": number (1 to horizon),
      "decay": "snap_back" | "linear" | "exponential"
    }
  ],
  "discrete_jumps": [
    {
      "month": number,
      "amount": number (-50000 to 50000, negative for expenses/losses),
      "variance": number (uncertainty around the amount)
    }
  ]
}

Parameter targets:
- mu_base: Base income level (multiplicative 0.7 = 30% income drop)
- sigma_base: Income volatility (multiplicative 1.3 = 30% more volatile)
- lambda: Jump frequency (multiplicative 1.5 = 50% more income shocks)
- expenses: Monthly fixed expenses (additive 200 = $200/month more expenses)

Common scenarios:
- Gas price spike: expenses additive +150-300, lambda multiplicative 1.2-1.5
- Recession: mu_base multiplicative 0.7-0.85, sigma_base multiplicative 1.2-1.4, duration 12-18 months
- Injury/illness: mu_base multiplicative 0.3-0.5 for 1-3 months, discrete_jump for medical bills
- Car breakdown: discrete_jump -1500 to -4000, variance 500
- Platform rate cut: mu_base multiplicative 0.8-0.9, permanent (long duration)
- Seasonal slowdown: mu_base multiplicative 0.7-0.85 for 2-3 months
- New competitor: lambda multiplicative 1.3, mu_base multiplicative 0.9

Always respond with valid JSON. Include a conversational "response" field explaining what you interpreted.`;

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
