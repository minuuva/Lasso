import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

// All required parameters for a complete simulation (11 total - no FICO)
const REQUIRED_PARAMS = {
  // Platform info
  platforms: {
    key: "platforms",
    label: "Applicant's Platforms",
    description: "Which gig platforms does the applicant work on?",
    type: "multi-select",
    options: ["Uber", "Lyft", "DoorDash", "Grubhub", "Instacart", "UberEats"],
    required: true,
  },
  hoursPerWeek: {
    key: "hoursPerWeek",
    label: "Weekly Hours",
    description: "How many hours per week does the applicant work?",
    type: "number",
    min: 5,
    max: 80,
    required: true,
  },
  monthsExperience: {
    key: "monthsExperience",
    label: "Gig Tenure",
    description: "How long has the applicant been doing gig work (in months)?",
    type: "number",
    min: 1,
    max: 120,
    required: true,
  },
  // Location
  metroArea: {
    key: "metroArea",
    label: "Work Location",
    description: "Which metro area does the applicant primarily work in?",
    type: "select",
    options: ["San Francisco", "New York", "Los Angeles", "Chicago", "Atlanta", "Dallas", "National Average", "Rural"],
    required: true,
  },
  // Personal situation
  hasVehicle: {
    key: "hasVehicle",
    label: "Vehicle Ownership",
    description: "Does the applicant own or lease a vehicle?",
    type: "boolean",
    required: true,
  },
  hasDependents: {
    key: "hasDependents",
    label: "Dependents",
    description: "Does the applicant have dependents?",
    type: "boolean",
    required: true,
  },
  // Financial situation
  liquidSavings: {
    key: "liquidSavings",
    label: "Liquid Savings",
    description: "What are the applicant's reported liquid savings?",
    type: "currency",
    min: 0,
    max: 100000,
    required: true,
  },
  monthlyExpenses: {
    key: "monthlyExpenses",
    label: "Monthly Expenses",
    description: "What are the applicant's monthly fixed expenses?",
    type: "currency",
    min: 500,
    max: 10000,
    required: true,
  },
  existingDebt: {
    key: "existingDebt",
    label: "Existing Debt",
    description: "What are the applicant's monthly debt obligations?",
    type: "currency",
    min: 0,
    max: 5000,
    required: true,
  },
  // Loan request
  loanAmount: {
    key: "loanAmount",
    label: "Requested Amount",
    description: "How much is the applicant requesting?",
    type: "currency",
    min: 500,
    max: 50000,
    required: true,
  },
  loanTermMonths: {
    key: "loanTermMonths",
    label: "Requested Term",
    description: "What loan term is the applicant requesting?",
    type: "select",
    options: ["6 months", "12 months", "18 months", "24 months", "36 months"],
    required: true,
  },
};

const SYSTEM_PROMPT = `You are a financial data extraction assistant for Lasso, a Monte Carlo credit risk assessment platform designed for loan officers at banks and financial institutions.

Your job is to extract loan applicant parameters from natural language input provided by loan officers. The loan officer is describing a gig worker applicant's situation and requesting a risk assessment.

IMPORTANT: We do NOT use FICO/credit scores. Our Monte Carlo simulation assesses risk based on gig work patterns and financial stability.

REQUIRED PARAMETERS TO EXTRACT (about the applicant):
1. platforms - Which gig platforms the applicant works on (uber, lyft, doordash, grubhub, instacart, ubereats)
2. hoursPerWeek - Total hours the applicant works per week across all platforms
3. monthsExperience - How long the applicant has been doing gig work (in months)
4. metroArea - Where the applicant primarily works (san_francisco, new_york, los_angeles, chicago, atlanta, dallas, national, rural)
5. hasVehicle - Does the applicant own/lease a vehicle (true/false)
6. hasDependents - Does the applicant support dependents (true/false)
7. liquidSavings - Applicant's reported emergency savings in dollars
8. monthlyExpenses - Applicant's monthly fixed expenses (rent, food, utilities) in dollars
9. existingDebt - Applicant's monthly debt payments in dollars
10. loanAmount - Requested loan amount
11. loanTermMonths - Requested repayment period in months

EXTRACTION RULES:
- Extract ONLY what is explicitly stated or strongly implied about the applicant
- Convert relative terms: "full-time" = 35-40 hours, "part-time" = 15-25 hours
- Convert time: "2 years" = 24 months, "a few months" = 3-6 months
- If a parameter is NOT mentioned or cannot be inferred, set it to null
- Be conservative - don't guess values
- Remember: the input is from a loan officer describing an applicant, not from the applicant themselves

RESPONSE FORMAT (JSON):
{
  "extractedParams": {
    "platforms": ["doordash", "uber"] or null,
    "hoursPerWeek": 30 or null,
    "monthsExperience": 12 or null,
    "metroArea": "san_francisco" or null,
    "hasVehicle": true or null,
    "hasDependents": false or null,
    "liquidSavings": 2000 or null,
    "monthlyExpenses": 1500 or null,
    "existingDebt": 200 or null,
    "loanAmount": 5000 or null,
    "loanTermMonths": 24 or null
  },
  "missingParams": ["list", "of", "missing", "parameter", "keys"],
  "confidence": "high" | "medium" | "low",
  "clarifyingNote": "Optional note about assumptions made or ambiguities in the applicant data"
}`;

export interface ExtractedParams {
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
  loanTermMonths: number | null;
}

export interface ExtractionResponse {
  extractedParams: ExtractedParams;
  missingParams: string[];
  confidence: "high" | "medium" | "low";
  clarifyingNote?: string;
  parameterDefinitions: typeof REQUIRED_PARAMS;
}

export async function POST(request: NextRequest) {
  try {
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        {
          error: "API key not configured",
          extractedParams: {},
          missingParams: Object.keys(REQUIRED_PARAMS),
          parameterDefinitions: REQUIRED_PARAMS,
        },
        { status: 503 }
      );
    }

    const body = await request.json();
    const { message, currentParams } = body;

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    // Build context from any existing params - only include non-null values
    let contextNote = "";
    if (currentParams && Object.keys(currentParams).length > 0) {
      const nonNullParams = Object.fromEntries(
        Object.entries(currentParams).filter(([, v]) => v !== null && v !== undefined)
      );
      if (Object.keys(nonNullParams).length > 0) {
        contextNote = `\n\nPREVIOUSLY EXTRACTED PARAMS (already collected - do not re-extract unless new info provided):\n${JSON.stringify(nonNullParams, null, 2)}`;
      }
    }

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: `Extract parameters from this loan officer's message about an applicant:

"${message}"${contextNote}

Return valid JSON only. Only extract NEW information - don't re-extract params already listed above.`,
        },
      ],
    });

    const textContent = response.content.find((c) => c.type === "text");
    if (!textContent || textContent.type !== "text") {
      throw new Error("No text response from AI");
    }

    // Parse the JSON response
    let extraction: ExtractionResponse;
    try {
      const jsonMatch = textContent.text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        // Merge with existing params - keep existing values, only add new ones
        const mergedParams: Record<string, unknown> = { ...currentParams };
        for (const [key, value] of Object.entries(parsed.extractedParams || {})) {
          // Only update if new value is not null and either old value is null or this is a new extraction
          if (value !== null && value !== undefined) {
            mergedParams[key] = value;
          }
        }

        // Recalculate missing params based on merged
        const missingParams = Object.keys(REQUIRED_PARAMS).filter(
          (key) => mergedParams[key] === null || mergedParams[key] === undefined
        );

        extraction = {
          extractedParams: mergedParams as ExtractedParams,
          missingParams,
          confidence: parsed.confidence || "medium",
          clarifyingNote: parsed.clarifyingNote,
          parameterDefinitions: REQUIRED_PARAMS,
        };
      } else {
        throw new Error("No JSON found in response");
      }
    } catch {
      // If parsing fails, return existing params unchanged
      extraction = {
        extractedParams: (currentParams || {}) as ExtractedParams,
        missingParams: Object.keys(REQUIRED_PARAMS).filter(
          (key) => !currentParams?.[key]
        ),
        confidence: "low",
        clarifyingNote: "Could not parse response",
        parameterDefinitions: REQUIRED_PARAMS,
      };
    }

    return NextResponse.json(extraction);
  } catch (error) {
    console.error("Parameter extraction error:", error);
    return NextResponse.json(
      {
        error: "Failed to extract parameters",
        details: String(error),
        parameterDefinitions: REQUIRED_PARAMS,
      },
      { status: 500 }
    );
  }
}
