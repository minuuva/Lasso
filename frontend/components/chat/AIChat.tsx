"use client";

import React, { useState, useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  scenario?: AIScenario | null;
  simulationResult?: SimulationResult | null;
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

interface SimulationResult {
  p_default: number;
  expected_loss: number;
  cvar_95: number;
  recommendation?: {
    approved: boolean;
    risk_tier: string;
    optimal_amount: number;
    reasoning: string[];
  };
}

const EXAMPLE_QUESTIONS = [
  "What if gas prices increase by 30%?",
  "What happens if I get injured for 2 months?",
  "How would a recession affect my loan risk?",
  "What if my car breaks down and I need $3000 for repairs?",
];

export function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm your Lasso AI assistant. Ask me 'what if' questions about financial scenarios, and I'll run Monte Carlo simulations to show you the impact on your loan risk. Try questions like 'What if gas prices spike 30%?' or 'What happens if I get injured?'",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
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
      // Step 1: Interpret the scenario with AI
      const interpretResponse = await fetch("/api/ai/interpret", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input.trim() }),
      });

      const interpretation = await interpretResponse.json();

      // Handle API errors with user-friendly messages
      if (!interpretResponse.ok || interpretation.error) {
        const errorMessage = interpretation.response || interpretation.error || "Failed to process your request";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessage.id
              ? {
                  ...m,
                  content: errorMessage,
                  isLoading: false,
                }
              : m
          )
        );
        setIsLoading(false);
        return;
      }

      // Step 2: If there's a scenario, run simulation
      let simulationResult = null;
      if (interpretation.should_run_simulation && interpretation.scenario) {
        // Update loading message
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessage.id
              ? { ...m, content: "Running Monte Carlo simulation..." }
              : m
          )
        );

        const simResponse = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            profile: {
              platforms: [{ name: "DoorDash", type: "delivery", tenure: 12 }],
              monthlyIncomeMean: 3500,
              incomeVolatility: 0.35,
              monthlyExpenses: 1800,
              savingsBuffer: 2000,
              ficoScore: 650,
            },
            config: {
              numPaths: 5000,
              horizonMonths: 24,
            },
            scenario: interpretation.scenario,
          }),
        });

        if (simResponse.ok) {
          const simData = await simResponse.json();
          simulationResult = simData.result || simData;
        }
      }

      // Build response message
      let responseContent = interpretation.response;

      if (simulationResult) {
        const pDefault = (simulationResult.p_default * 100).toFixed(1);
        const expectedLoss = simulationResult.expected_loss?.toFixed(0) || "N/A";

        responseContent += `\n\n**Simulation Results:**\n`;
        responseContent += `- Default Probability: ${pDefault}%\n`;
        responseContent += `- Expected Loss: $${expectedLoss}\n`;

        if (simulationResult.recommendation) {
          responseContent += `- Risk Tier: ${simulationResult.recommendation.risk_tier}\n`;
          responseContent += `- Loan ${simulationResult.recommendation.approved ? "Approved" : "Declined"}`;
        }
      }

      // Replace loading message with actual response
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? {
                ...m,
                content: responseContent,
                scenario: interpretation.scenario,
                simulationResult,
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
                content: "Sorry, I encountered an error processing your request. Please try again.",
                isLoading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (question: string) => {
    setInput(question);
  };

  return (
    <div className="flex flex-col h-full bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
      {/* Header */}
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
            <h3 className="font-display font-semibold text-white/90">
              Lasso AI
            </h3>
            <p className="text-xs text-white/40">
              Scenario Analysis & Risk Simulation
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
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
                <div className="text-sm whitespace-pre-wrap">
                  {message.content}
                </div>
              )}

              {/* Scenario badge */}
              {message.scenario && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <span className="text-[10px] uppercase tracking-wider text-white/40">
                    Scenario Applied
                  </span>
                  <p className="text-xs text-amber-400/80 mt-1">
                    {message.scenario.narrative}
                  </p>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Example questions */}
      {messages.length <= 1 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-white/40 mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => handleExampleClick(q)}
                className="text-xs px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-white/60 hover:bg-white/10 hover:text-white/80 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a 'what if' scenario..."
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
