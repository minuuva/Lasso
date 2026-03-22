"use client";

import React, { useState, useRef, useEffect } from "react";
import { useSimulationOptional } from "@/contexts/SimulationContext";
import {
  BASELINE_SIMULATION_QUERY,
  chartPathToProxyUrl,
  gigProfileToAiPayload,
  type AiLayerSimulateResponse,
} from "@/lib/aiLayer";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  scenario?: AIScenario | null;
  aiSimulation?: AiLayerSimulateResponse | null;
  isLoading?: boolean;
}

interface AIScenario {
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
}

const EXAMPLE_QUESTIONS = [
  "What if gas prices increase by 30%?",
  "Model recession_2020 starting at month 3",
  "What happens if I can't work for 2 months?",
  "Severe gas spike for 6 months from month 0",
];

function formatMetrics(data: AiLayerSimulateResponse): string {
  const m = data.metrics;
  if (!m) return "";
  return [
    `Default probability: ${(m.p_default * 100).toFixed(1)}%`,
    `Expected loss: $${Math.round(m.expected_loss).toLocaleString()}`,
    `CVaR (95%): $${Math.round(m.cvar_95).toLocaleString()}`,
    `Risk tier: ${m.risk_tier}`,
    `Approved: ${m.approved ? "yes" : "no"}`,
  ].join("\n");
}

export function AIChat() {
  const sim = useSimulationOptional();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Ask a “what if” scenario. I’ll translate it into income/expense shocks and run the Monte Carlo engine (no credit score). Try naming catalog shocks like recession_2020 or gas_spike_severe, or describe any custom event.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const text = input.trim();
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };
    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "Analyzing scenario...",
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const { userData, loanPreferences } = sim
        ? gigProfileToAiPayload(sim.state.profile, sim.state.loanParams)
        : {
            userData: {
              platforms: ["doordash"],
              hours_per_week: 40,
              monthly_income_estimate: Math.round(40 * 4.33 * 22),
              metro_area: "national",
              months_as_gig_worker: 12,
              has_vehicle: true,
              has_dependents: false,
              liquid_savings: 2000,
              monthly_fixed_expenses: 1800,
              existing_debt_obligations: 0,
            },
            loanPreferences: { amount: 5000, term_months: 24, max_rate: 0.2 },
          };

      const interpretResponse = await fetch("/api/ai/interpret", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          context: userData,
        }),
      });

      const interpretation = await interpretResponse.json();

      if (!interpretResponse.ok || interpretation.error) {
        const err =
          interpretation.response ||
          interpretation.error ||
          "Failed to process your request";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessage.id
              ? { ...m, content: err, isLoading: false }
              : m
          )
        );
        return;
      }

      let aiSimulation: AiLayerSimulateResponse | null = null;
      let assistantBody = interpretation.response as string;

      if (interpretation.should_run_simulation && interpretation.scenario) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessage.id
              ? { ...m, content: "Running Monte Carlo simulation...", isLoading: true }
              : m
          )
        );

        const stressQuery = `Stress scenario for gig worker: ${interpretation.scenario.narrative || text}. Apply structured shocks.`;
        const query =
          stressQuery.length >= 10
            ? stressQuery
            : `${stressQuery} Repeat for full risk assessment.`;

        const simResponse = await fetch("/api/simulation/ai-layer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: interpretation.scenario ? query : BASELINE_SIMULATION_QUERY,
            userData,
            loanPreferences,
            structuredScenario: interpretation.scenario,
            generateCharts: true,
          }),
        });

        aiSimulation = (await simResponse.json()) as AiLayerSimulateResponse;

        if (!simResponse.ok) {
          const detail =
            typeof aiSimulation.detail === "string"
              ? aiSimulation.detail
              : (aiSimulation as { error?: string }).error;
          assistantBody += `\n\n**Simulation error:** ${detail || simResponse.statusText}\n\nSet AI_MODEL_API_BASE_URL in .env.local and run the Python FastAPI server.`;
        } else {
          const summaryBlock =
            aiSimulation.quick_summary ||
            (aiSimulation.summary
              ? aiSimulation.summary.slice(0, 600)
              : "");
          assistantBody += summaryBlock ? `\n\n${summaryBlock}` : "";
          const metricsText = formatMetrics(aiSimulation);
          if (metricsText) assistantBody += `\n\n${metricsText}`;
        }
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? {
                ...m,
                content: assistantBody,
                scenario: interpretation.scenario,
                aiSimulation,
                isLoading: false,
              }
            : m
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? {
                ...m,
                content: "Sorry, something went wrong. Try again.",
                isLoading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
      <div className="px-6 py-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
          </div>
          <div>
            <h3 className="font-display font-semibold text-white/90">Lasso AI</h3>
            <p className="text-xs text-white/40">Scenarios & Monte Carlo</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                message.role === "user"
                  ? "bg-amber-500/20 border border-amber-500/30 text-white/90"
                  : "bg-white/5 border border-white/10 text-white/80"
              }`}
            >
              {message.isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  <span className="text-sm">{message.content}</span>
                </div>
              ) : (
                <>
                  <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                  {message.scenario && (
                    <div className="mt-3 pt-3 border-t border-white/10">
                      <span className="text-[10px] uppercase tracking-wider text-white/40">
                        Scenario overlay
                      </span>
                      <p className="text-xs text-amber-400/80 mt-1">
                        {message.scenario.narrative}
                      </p>
                    </div>
                  )}
                  {message.aiSimulation?.charts && message.aiSimulation.charts.length > 0 && (
                    <div className="mt-4 space-y-3 max-h-[320px] overflow-y-auto">
                      {message.aiSimulation.charts.map((c) => (
                        <div key={`${c.type}-${c.path}`} className="rounded-lg overflow-hidden border border-white/10">
                          <div className="text-[10px] text-white/40 px-2 py-1 bg-white/5">
                            {c.description}
                          </div>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={chartPathToProxyUrl(c.path)}
                            alt={c.description}
                            className="w-full h-auto max-h-48 object-contain bg-black/40"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {messages.length <= 1 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-white/40 mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => setInput(q)}
                className="text-xs px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-white/60 hover:bg-white/10 hover:text-white/80 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe a shock or ‘what if’ event..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white/90 placeholder-white/30 focus:outline-none focus:border-amber-500/50 transition-colors"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-amber-500 hover:bg-amber-400 disabled:bg-white/10 disabled:text-white/30 text-white font-medium text-sm rounded-xl transition-colors"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              "Send"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default AIChat;
