"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";

// Simple unique ID generator
let idCounter = 0;
function generateId(): string {
  return `msg_${Date.now()}_${++idCounter}`;
}

// Consistent number formatting (avoids hydration mismatch from toLocaleString)
function formatNumber(num: number): string {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isLoading?: boolean;
  questions?: ParameterQuestion[];
  extractedParams?: Record<string, unknown>;
}

interface ParameterQuestion {
  key: string;
  label: string;
  description: string;
  type: "select" | "multi-select" | "number" | "currency" | "boolean";
  options?: string[];
  min?: number;
  max?: number;
}

interface Chat {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messages: Message[];
}

interface ExtractedParams {
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

// Loan officer user (the actual user of this tool)
const LOAN_OFFICER = {
  name: "Sarah Chen",
  role: "Senior Loan Officer",
  avatar: "SC",
  institution: "Capital One",
};

const INITIAL_CHATS: Chat[] = [];

// Example applicant profiles for quick assessment
const APPLICANT_EXAMPLES = [
  {
    icon: "rideshare",
    title: "Rideshare Driver",
    description: "Full-time Uber/Lyft driver in SF requesting $5,000",
  },
  {
    icon: "delivery",
    title: "Delivery Partner",
    description: "DoorDash driver, 25 hrs/week, needs $3,000 for vehicle repairs",
  },
  {
    icon: "multiplatform",
    title: "Multi-Platform Worker",
    description: "Works Instacart and UberEats part-time, requesting small personal loan",
  },
  {
    icon: "newworker",
    title: "New Gig Worker",
    description: "Started Lyft 3 months ago, limited history, requesting $2,000",
  },
];

function ApplicantIcon({ type }: { type: string }) {
  const iconClass = "w-5 h-5";
  switch (type) {
    case "rideshare":
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 18.75a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 0 1-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m3 0h1.125c.621 0 1.129-.504 1.09-1.124a17.902 17.902 0 0 0-3.213-9.193 2.056 2.056 0 0 0-1.58-.86H14.25M16.5 18.75h-2.25m0-11.177v-.958c0-.568-.422-1.048-.987-1.106a48.554 48.554 0 0 0-10.026 0 1.106 1.106 0 0 0-.987 1.106v7.635m12-6.677v6.677m0 4.5v-4.5m0 0h-12" />
        </svg>
      );
    case "delivery":
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5V6a3.75 3.75 0 1 0-7.5 0v4.5m11.356-1.993 1.263 12c.07.665-.45 1.243-1.119 1.243H4.25a1.125 1.125 0 0 1-1.12-1.243l1.264-12A1.125 1.125 0 0 1 5.513 7.5h12.974c.576 0 1.059.435 1.119 1.007ZM8.625 10.5a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm7.5 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" />
        </svg>
      );
    case "multiplatform":
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 6.878V6a2.25 2.25 0 0 1 2.25-2.25h7.5A2.25 2.25 0 0 1 18 6v.878m-12 0c.235-.083.487-.128.75-.128h10.5c.263 0 .515.045.75.128m-12 0A2.25 2.25 0 0 0 4.5 9v.878m13.5-3A2.25 2.25 0 0 1 19.5 9v.878m0 0a2.246 2.246 0 0 0-.75-.128H5.25c-.263 0-.515.045-.75.128m15 0A2.25 2.25 0 0 1 21 12v6a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 18v-6c0-.98.626-1.813 1.5-2.122" />
        </svg>
      );
    case "newworker":
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
        </svg>
      );
    default:
      return null;
  }
}

// Individual question input (no confirm button - just updates state)
function QuestionInput({
  question,
  value,
  onChange,
}: {
  question: ParameterQuestion;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <div className="bg-white/[0.04] border border-white/[0.08] rounded-xl p-4 mb-3">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-sm font-medium text-white/90 mb-1">{question.label}</div>
          <div className="text-xs text-white/50">{question.description}</div>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
          Required
        </span>
      </div>

      {question.type === "select" && question.options && (
        <div className="grid grid-cols-2 gap-2 mt-3">
          {question.options.map((option) => (
            <button
              key={`${question.key}-${option}`}
              type="button"
              onClick={() => onChange(option)}
              className={`px-3 py-2 text-sm rounded-lg border transition-all ${
                value === option
                  ? "bg-amber-500/20 border-amber-500/50 text-amber-400"
                  : "bg-white/[0.03] border-white/[0.08] text-white/70 hover:bg-white/[0.06] hover:border-white/[0.12]"
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {question.type === "multi-select" && question.options && (
        <div className="flex flex-wrap gap-2 mt-3">
          {question.options.map((option) => {
            const selected = Array.isArray(value) && value.includes(option);
            return (
              <button
                key={`${question.key}-${option}`}
                type="button"
                onClick={() => {
                  const current = Array.isArray(value) ? value : [];
                  const newValue = selected
                    ? current.filter((v) => v !== option)
                    : [...current, option];
                  onChange(newValue);
                }}
                className={`px-3 py-1.5 text-sm rounded-lg border transition-all ${
                  selected
                    ? "bg-amber-500/20 border-amber-500/50 text-amber-400"
                    : "bg-white/[0.03] border-white/[0.08] text-white/70 hover:bg-white/[0.06]"
                }`}
              >
                {option}
              </button>
            );
          })}
        </div>
      )}

      {question.type === "boolean" && (
        <div className="flex gap-3 mt-3">
          {[
            { label: "Yes", val: true },
            { label: "No", val: false },
          ].map((option) => (
            <button
              key={`${question.key}-${option.label}`}
              type="button"
              onClick={() => onChange(option.val)}
              className={`flex-1 px-4 py-2 text-sm rounded-lg border transition-all ${
                value === option.val
                  ? "bg-amber-500/20 border-amber-500/50 text-amber-400"
                  : "bg-white/[0.03] border-white/[0.08] text-white/70 hover:bg-white/[0.06]"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}

      {(question.type === "number" || question.type === "currency") && (
        <div className="mt-3">
          <div className="relative">
            {question.type === "currency" && (
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40">$</span>
            )}
            <input
              type="text"
              inputMode="numeric"
              value={
                value
                  ? question.type === "currency"
                    ? formatNumber(Number(value))
                    : question.key === "monthsExperience"
                    ? `${formatNumber(Number(value))} months`
                    : formatNumber(Number(value))
                  : ""
              }
              onChange={(e) => {
                const raw = e.target.value.replace(/[^0-9]/g, "");
                onChange(raw ? Number(raw) : null);
              }}
              className={`w-full bg-white/[0.05] border border-white/[0.1] rounded-lg py-2 text-sm text-white placeholder-white/30 focus:outline-none focus:border-amber-500/50 ${
                question.type === "currency" ? "pl-7 pr-3" : "px-3"
              }`}
              placeholder={question.type === "currency" ? "0" : question.key === "monthsExperience" ? "e.g. 12 months" : "Enter value"}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Progress indicator for data collection
function DataCollectionProgress({ collected, total }: { collected: number; total: number }) {
  const percentage = Math.round((collected / total) * 100);
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="flex-1 h-1.5 bg-white/[0.08] rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-white/50 font-medium">
        {collected}/{total} data points
      </span>
    </div>
  );
}

export default function SimulateHub() {
  const [chats, setChats] = useState<Chat[]>(INITIAL_CHATS);
  const [activeChat, setActiveChat] = useState<Chat | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [extractedParams, setExtractedParams] = useState<Partial<ExtractedParams>>({});
  const [pendingQuestions, setPendingQuestions] = useState<ParameterQuestion[]>([]);
  const [pendingAnswers, setPendingAnswers] = useState<Record<string, unknown>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 11 total params (no FICO)
  const totalParams = 11;
  const collectedParams = Object.values(extractedParams).filter(
    (v) => v !== null && v !== undefined
  ).length;

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleNewAssessment = () => {
    setActiveChat(null);
    setMessages([]);
    setExtractedParams({});
    setPendingQuestions([]);
    setPendingAnswers({});
  };

  const handleSelectChat = (chat: Chat) => {
    setActiveChat(chat);
    setMessages(chat.messages);
  };

  // Handle updating a pending answer
  const handleAnswerChange = (key: string, value: unknown) => {
    setPendingAnswers((prev) => ({ ...prev, [key]: value }));
  };

  // Submit all pending answers at once
  const handleSubmitAnswers = () => {
    const newParams = { ...extractedParams };

    // Merge all pending answers into params
    for (const [key, value] of Object.entries(pendingAnswers)) {
      if (value !== null && value !== undefined && value !== "") {
        (newParams as Record<string, unknown>)[key] = value;
      }
    }

    setExtractedParams(newParams);

    // Create a summary message of what was answered
    const answeredKeys = Object.keys(pendingAnswers).filter(
      (k) => pendingAnswers[k] !== null && pendingAnswers[k] !== undefined
    );

    if (answeredKeys.length > 0) {
      const answerMessage: Message = {
        id: generateId(),
        role: "user",
        content: answeredKeys.map((k) => `${k}: ${JSON.stringify(pendingAnswers[k])}`).join("\n"),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, answerMessage]);
    }

    // Clear pending
    setPendingQuestions([]);
    setPendingAnswers({});

    // Check if all params are now collected
    const filledCount = Object.values(newParams).filter(
      (v) => v !== null && v !== undefined
    ).length;

    if (filledCount >= totalParams) {
      outputCompleteData(newParams as ExtractedParams);
    } else {
      // Still need more data - fetch remaining questions
      fetchRemainingQuestions(newParams);
    }
  };

  // Fetch remaining questions after submitting some answers
  const fetchRemainingQuestions = async (currentParams: Partial<ExtractedParams>) => {
    const loadingMessage: Message = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true,
    };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      const extractResponse = await fetch("/api/ai/extract-params", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: "Continue collecting remaining parameters",
          currentParams,
        }),
      });

      const extraction = await extractResponse.json();
      const missingParams = extraction.missingParams || [];
      const paramDefs = extraction.parameterDefinitions || {};

      if (missingParams.length > 0) {
        const questions: ParameterQuestion[] = missingParams.slice(0, 6).map((key: string) => {
          const def = paramDefs[key] || {};
          return {
            key,
            label: def.label || key,
            description: def.description || `Please provide ${key}`,
            type: def.type || "text",
            options: def.options,
            min: def.min,
            max: def.max,
          };
        });

        setPendingQuestions(questions);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessage.id
              ? {
                  ...m,
                  content: `Please provide the remaining applicant information (${missingParams.length} fields):`,
                  isLoading: false,
                  questions,
                }
              : m
          )
        );
      } else {
        outputCompleteData(currentParams as ExtractedParams);
        setMessages((prev) => prev.filter((m) => m.id !== loadingMessage.id));
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? { ...m, content: "Error fetching questions. Please try again.", isLoading: false }
            : m
        )
      );
    }
  };

  // When all params collected, output in Python CustomerApplication format
  const outputCompleteData = (params: ExtractedParams) => {
    // Convert loan term to number if string
    const termMonthsMap: Record<string, number> = {
      "6 months": 6,
      "12 months": 12,
      "18 months": 18,
      "24 months": 24,
      "36 months": 36,
    };
    const termMonths = typeof params.loanTermMonths === "string"
      ? termMonthsMap[params.loanTermMonths] || 24
      : params.loanTermMonths;

    // Convert metro area to snake_case
    const metroAreaMap: Record<string, string> = {
      "San Francisco": "san_francisco",
      "New York": "new_york",
      "Los Angeles": "los_angeles",
      "Chicago": "chicago",
      "Atlanta": "atlanta",
      "Dallas": "dallas",
      "National Average": "national",
      "Rural": "rural",
    };
    const metroArea = params.metroArea ? metroAreaMap[params.metroArea] || params.metroArea.toLowerCase().replace(/\s+/g, "_") : "national";

    // Build platforms_and_hours array: [(platform, hours/week, tenure_months), ...]
    // Since we collect total hours, distribute evenly across platforms
    const platforms = params.platforms || [];
    const hoursPerPlatform = platforms.length > 0 ? (params.hoursPerWeek || 0) / platforms.length : 0;
    const tenure = params.monthsExperience || 0;

    const platformsAndHours = platforms.map((p) => [
      p.toLowerCase(),
      hoursPerPlatform,
      tenure,
    ]);

    // Build the CustomerApplication format
    const customerApplication = {
      platforms_and_hours: platformsAndHours,
      metro_area: metroArea,
      months_as_gig_worker: params.monthsExperience,
      has_vehicle: params.hasVehicle,
      has_dependents: params.hasDependents,
      liquid_savings: params.liquidSavings,
      monthly_fixed_expenses: params.monthlyExpenses,
      existing_debt_obligations: params.existingDebt,
      loan_request_amount: params.loanAmount,
      requested_term_months: termMonths,
      acceptable_rate_range: [0.08, 0.20],
    };

    // Log to console for your team to pick up
    console.log("=== CUSTOMER APPLICATION ===");
    console.log(JSON.stringify(customerApplication, null, 2));
    console.log("============================");

    // Format for display (Python-like)
    const pythonFormat = `CustomerApplication(
    platforms_and_hours=[
${platformsAndHours.map((p) => `        ('${p[0]}', ${p[1].toFixed(1)}, ${p[2]})`).join(",\n")}
    ],
    metro_area='${metroArea}',
    months_as_gig_worker=${params.monthsExperience},
    has_vehicle=${params.hasVehicle ? "True" : "False"},
    has_dependents=${params.hasDependents ? "True" : "False"},
    liquid_savings=${params.liquidSavings},
    monthly_fixed_expenses=${params.monthlyExpenses},
    existing_debt_obligations=${params.existingDebt},
    loan_request_amount=${params.loanAmount},
    requested_term_months=${termMonths},
    acceptable_rate_range=(0.08, 0.20)
)`;

    const successMessage: Message = {
      id: generateId(),
      role: "assistant",
      content: `## All Data Collected

All 11 required data points have been gathered. The application bundle is ready for AI Layer processing.

\`\`\`python
${pythonFormat}
\`\`\`

**Ready for AI Layer & Monte Carlo simulation.**`,
      timestamp: new Date(),
      extractedParams: customerApplication as unknown as Record<string, unknown>,
    };

    setMessages((prev) => [...prev, successMessage]);
  };

  const handleSubmit = async (e?: React.FormEvent, suggestionText?: string) => {
    e?.preventDefault();
    const messageText = suggestionText || input.trim();
    if (!messageText || isLoading) return;

    const userMessageId = generateId();
    const loadingMessageId = generateId();

    const userMessage: Message = {
      id: userMessageId,
      role: "user",
      content: messageText,
      timestamp: new Date(),
    };

    const loadingMessage: Message = {
      id: loadingMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const extractResponse = await fetch("/api/ai/extract-params", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: messageText,
          currentParams: extractedParams,
        }),
      });

      const extraction = await extractResponse.json();

      if (extraction.error && !extraction.extractedParams) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessageId
              ? {
                  ...m,
                  content: "I need more details about the applicant. Please provide their gig work information, income details, or loan request.",
                  isLoading: false,
                }
              : m
          )
        );
        setIsLoading(false);
        return;
      }

      const newParams = { ...extractedParams, ...extraction.extractedParams };
      setExtractedParams(newParams);

      let responseContent = "";
      const extractedCount = Object.values(newParams).filter(
        (v) => v !== null && v !== undefined
      ).length;

      if (extraction.clarifyingNote) {
        responseContent += `${extraction.clarifyingNote}\n\n`;
      }

      if (extractedCount > 0) {
        responseContent += "**Applicant data captured:**\n";
        if (newParams.platforms?.length) {
          responseContent += `- Platforms: ${newParams.platforms.join(", ")}\n`;
        }
        if (newParams.hoursPerWeek) {
          responseContent += `- Weekly hours: ${newParams.hoursPerWeek}\n`;
        }
        if (newParams.metroArea) {
          responseContent += `- Metro area: ${newParams.metroArea}\n`;
        }
        if (newParams.loanAmount) {
          responseContent += `- Requested amount: $${formatNumber(newParams.loanAmount)}\n`;
        }
        if (newParams.monthsExperience) {
          responseContent += `- Gig tenure: ${newParams.monthsExperience} months\n`;
        }
        if (newParams.loanTermMonths) {
          responseContent += `- Loan term: ${newParams.loanTermMonths} months\n`;
        }
        responseContent += "\n";
      }

      // Only get ACTUALLY missing params
      const missingParams = extraction.missingParams || [];
      const paramDefs = extraction.parameterDefinitions || {};

      if (missingParams.length > 0) {
        responseContent += `Please provide the remaining applicant information (${missingParams.length} fields):`;

        // Create questions for all missing params (show up to 6 at a time)
        const questions: ParameterQuestion[] = missingParams.slice(0, 6).map((key: string) => {
          const def = paramDefs[key] || {};
          return {
            key,
            label: def.label || key,
            description: def.description || `Please provide ${key}`,
            type: def.type || "text",
            options: def.options,
            min: def.min,
            max: def.max,
          };
        });

        setPendingQuestions(questions);
        setPendingAnswers({});

        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessageId
              ? {
                  ...m,
                  content: responseContent,
                  isLoading: false,
                  questions,
                  extractedParams: newParams,
                }
              : m
          )
        );
      } else {
        responseContent += "All required data collected!\n";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMessageId
              ? {
                  ...m,
                  content: responseContent,
                  isLoading: false,
                }
              : m
          )
        );

        outputCompleteData(newParams as ExtractedParams);
      }

      if (!activeChat) {
        const newChat: Chat = {
          id: generateId(),
          title: `Application: ${newParams.platforms?.join("/") || "New"} - $${newParams.loanAmount ? formatNumber(newParams.loanAmount) : "TBD"}`,
          lastMessage: responseContent.slice(0, 60) + "...",
          timestamp: new Date(),
          messages: [],
        };
        setChats((prev) => [newChat, ...prev]);
        setActiveChat(newChat);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessageId
            ? {
                ...m,
                content: "Error processing request. Please try again.",
                isLoading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return "Just now";
  };

  // Check if all pending questions have been answered
  const allQuestionsAnswered = pendingQuestions.length > 0 &&
    pendingQuestions.every((q) => {
      const val = pendingAnswers[q.key];
      if (q.type === "multi-select") {
        return Array.isArray(val) && val.length > 0;
      }
      return val !== null && val !== undefined && val !== "";
    });

  const showWelcome = messages.length === 0;

  return (
    <div className="flex h-screen bg-[#0a0a0f] text-white overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? "w-72" : "w-0"
        } flex-shrink-0 border-r border-white/[0.06] bg-[#08080c] transition-all duration-300 overflow-hidden flex flex-col`}
      >
        <div className="p-4 border-b border-white/[0.06]">
          <button
            onClick={handleNewAssessment}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-white/[0.05] hover:bg-white/[0.08] border border-white/[0.08] hover:border-white/[0.12] transition-all duration-200 text-sm font-medium text-white/80 hover:text-white"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Assessment
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide p-3 space-y-1">
          <div className="text-[10px] font-medium uppercase tracking-wider text-white/30 px-3 py-2">
            Recent Applications
          </div>
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => handleSelectChat(chat)}
              className={`w-full text-left p-3 rounded-lg transition-all duration-200 group ${
                activeChat?.id === chat.id
                  ? "bg-white/[0.08] border border-white/[0.1]"
                  : "hover:bg-white/[0.04] border border-transparent"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-sm text-white/80 font-medium truncate">{chat.title}</span>
                <span className="text-[10px] text-white/30 flex-shrink-0">{formatTime(chat.timestamp)}</span>
              </div>
              <p className="text-xs text-white/40 mt-1 truncate">{chat.lastMessage}</p>
            </button>
          ))}
          {chats.length === 0 && (
            <div className="px-3 py-6 text-center">
              <p className="text-xs text-white/30">No recent assessments</p>
            </div>
          )}
        </div>

        {/* Loan Officer Profile */}
        <div className="p-4 border-t border-white/[0.06]">
          <div className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.03]">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center text-sm font-semibold text-white">
              {LOAN_OFFICER.avatar}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-white/90 truncate">{LOAN_OFFICER.name}</div>
              <div className="text-[10px] text-white/40">{LOAN_OFFICER.role}</div>
            </div>
          </div>
          <div className="mt-2 px-3">
            <div className="text-[10px] text-white/30">{LOAN_OFFICER.institution}</div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-white/[0.05] transition-colors"
            >
              <svg className="w-5 h-5 text-white/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
              </svg>
            </button>
            <Link href="/" className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-400 via-orange-500 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
                <span className="text-white font-bold text-xs">L</span>
              </div>
              <span className="text-white font-display font-bold text-lg tracking-tight">Lasso</span>
            </Link>
            <span className="hidden md:inline-block text-xs text-white/30 px-2 py-1 rounded bg-white/[0.05]">
              Risk Assessment Console
            </span>
          </div>

          {collectedParams > 0 && (
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.05] border border-white/[0.08]">
              <div className="w-2 h-2 rounded-full bg-amber-400" />
              <span className="text-xs text-white/60">{collectedParams}/{totalParams} data points</span>
            </div>
          )}

          <div className="flex items-center gap-2">
            <button className="px-4 py-2 rounded-lg text-sm text-white/60 hover:text-white/80 hover:bg-white/[0.05] transition-all">
              Documentation
            </button>
          </div>
        </header>

        <div
          className="flex-1 min-h-0 overflow-y-auto scrollbar-hide"
          onWheel={(e) => {
            // Explicitly handle wheel scroll to bypass any preventDefault
            e.currentTarget.scrollTop += e.deltaY;
          }}
        >
          {showWelcome ? (
            <div className="min-h-full flex flex-col items-center justify-center px-6 py-12">
              <div className="max-w-2xl w-full text-center">
                <div className="mb-3">
                  <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                    </svg>
                    Monte Carlo Risk Engine
                  </span>
                </div>

                <h1 className="font-display text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
                  Assess a{" "}
                  <span className="text-gradient-accent">gig worker</span> application
                </h1>

                <p className="text-lg text-white/40 mb-12 max-w-lg mx-auto">
                  Enter applicant details to collect data for Monte Carlo risk simulation.
                </p>

                <div className="grid grid-cols-2 gap-3 mb-12">
                  {APPLICANT_EXAMPLES.map((example) => (
                    <button
                      key={example.title}
                      onClick={() => handleSubmit(undefined, example.description)}
                      className="group p-4 rounded-xl bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] hover:border-white/[0.1] transition-all duration-200 text-left"
                    >
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-white/[0.05] group-hover:bg-amber-500/10 flex items-center justify-center text-white/40 group-hover:text-amber-400 transition-colors">
                          <ApplicantIcon type={example.icon} />
                        </div>
                        <div>
                          <div className="text-sm font-medium text-white/80 group-hover:text-white mb-0.5">
                            {example.title}
                          </div>
                          <div className="text-xs text-white/40">{example.description}</div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                <div className="text-xs text-white/30 flex items-center justify-center gap-4">
                  <span>Powered by 5,000-path Monte Carlo simulation</span>
                  <span className="w-1 h-1 rounded-full bg-white/20" />
                  <span>JPMorgan Chase calibration data</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-6 py-8">
              {collectedParams > 0 && (
                <DataCollectionProgress collected={collectedParams} total={totalParams} />
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`mb-6 ${message.role === "user" ? "flex justify-end" : ""}`}
                >
                  {message.role === "assistant" ? (
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500 flex-shrink-0 flex items-center justify-center">
                        <span className="text-white text-xs font-bold">L</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        {message.isLoading ? (
                          <div className="flex items-center gap-2 py-2">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                              <div className="w-2 h-2 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                            </div>
                            <span className="text-sm text-white/50">Processing...</span>
                          </div>
                        ) : (
                          <>
                            <div className="prose prose-invert prose-sm max-w-none">
                              <div className="text-white/80 whitespace-pre-wrap leading-relaxed">
                                {message.content}
                              </div>
                            </div>

                            {message.questions && message.questions.length > 0 && pendingQuestions.length > 0 && (
                              <div className="mt-4">
                                {message.questions.map((q) => (
                                  <QuestionInput
                                    key={`question-${q.key}`}
                                    question={q}
                                    value={pendingAnswers[q.key]}
                                    onChange={(val) => handleAnswerChange(q.key, val)}
                                  />
                                ))}

                                {/* Single confirm button at bottom */}
                                <button
                                  onClick={handleSubmitAnswers}
                                  disabled={!allQuestionsAnswered}
                                  className="w-full mt-4 px-6 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:bg-white/10 disabled:text-white/30 text-white font-semibold transition-all disabled:cursor-not-allowed"
                                >
                                  Continue
                                </button>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-[80%] px-4 py-3 rounded-2xl bg-blue-500/20 border border-blue-500/30">
                      <div className="text-sm text-white/90 whitespace-pre-wrap">{message.content}</div>
                    </div>
                  )}
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="flex-shrink-0 border-t border-white/[0.06] p-4 bg-[#0a0a0f]">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder={pendingQuestions.length > 0 ? "Or type additional info here..." : "Describe the loan applicant..."}
                rows={1}
                className="w-full bg-white/[0.05] border border-white/[0.08] rounded-xl px-4 py-3 pr-12 text-sm text-white placeholder-white/30 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 resize-none transition-all"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-amber-500 hover:bg-amber-400 disabled:bg-white/10 disabled:text-white/30 text-white transition-colors disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
                  </svg>
                )}
              </button>
            </div>
            <div className="flex items-center justify-center mt-2 text-[10px] text-white/25">
              Data collection for Monte Carlo risk simulation.
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
