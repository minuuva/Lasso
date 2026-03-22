"use client";

import React, { useEffect, useRef, useState, useMemo } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { ARCHETYPES } from "@/lib/constants";
import { generateMockSimulationResult } from "@/mocks/generators";
import { archetypeToProfile } from "@/mocks/generators";

gsap.registerPlugin(ScrollTrigger);

// Mini archetype cards for selection
const PREVIEW_ARCHETYPES = ARCHETYPES.slice(0, 4); // First 4 archetypes

export function PreviewSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const [selectedArchetype, setSelectedArchetype] = useState(PREVIEW_ARCHETYPES[0]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Only run on client to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Generate simulation data when archetype changes - only on client
  const simulationData = useMemo(() => {
    if (!mounted) return null;
    const profile = archetypeToProfile(selectedArchetype);
    const result = generateMockSimulationResult(profile, 24);
    return result;
  }, [selectedArchetype, mounted]);

  // Format data for Recharts
  const chartData = useMemo(() => {
    if (!simulationData) return [];

    return simulationData.incomePercentiles.p50.map((_, index) => ({
      month: index,
      p10: simulationData.incomePercentiles.p10[index],
      p25: simulationData.incomePercentiles.p25[index],
      p50: simulationData.incomePercentiles.p50[index],
      p75: simulationData.incomePercentiles.p75[index],
      p90: simulationData.incomePercentiles.p90[index],
    }));
  }, [simulationData]);

  // Handle archetype selection
  const handleSelectArchetype = (archetype: typeof selectedArchetype) => {
    if (archetype.id === selectedArchetype.id) return;
    setIsAnimating(true);
    setTimeout(() => {
      setSelectedArchetype(archetype);
      setIsAnimating(false);
    }, 300);
  };

  // FICO comparison result
  const riskAssessment = useMemo(() => {
    const delta = selectedArchetype.defaultRiskCategory;
    if (delta === "low") return { label: "FICO Overestimates Risk", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20" };
    if (delta === "high") return { label: "FICO Underestimates Risk", color: "text-red-400", bg: "bg-red-500/10 border-red-500/20" };
    return { label: "Aligned with FICO", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" };
  }, [selectedArchetype]);

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      gsap.from(".preview-content", {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 60%",
        },
        opacity: 0,
        y: 60,
        duration: 1,
        ease: "power3.out",
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full py-24 px-8 md:px-16 lg:px-24"
    >
      <div className="preview-content">
        {/* Section label */}
        <div className="mb-8">
          <span className="inline-flex items-center gap-2 text-[11px] font-medium tracking-[0.25em] uppercase text-white/30">
            <span className="w-8 h-px bg-white/20" />
            Live Preview
          </span>
        </div>

        {/* Headline */}
        <h2 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight mb-6">
          <span className="text-white/90">See it in action.</span>
        </h2>

        <p className="max-w-xl text-lg text-white/50 leading-relaxed mb-12">
          Select a gig worker persona and watch the simulation reveal their true risk profile.
        </p>

        {/* Main content grid */}
        <div className="grid lg:grid-cols-12 gap-8">
          {/* Archetype selector */}
          <div className="lg:col-span-4 space-y-4">
            <h3 className="text-sm font-medium text-white/50 mb-4">Select Persona</h3>
            {PREVIEW_ARCHETYPES.map((archetype) => (
              <button
                key={archetype.id}
                onClick={() => handleSelectArchetype(archetype)}
                className={`w-full text-left p-4 rounded-xl border transition-all duration-300 ${
                  selectedArchetype.id === archetype.id
                    ? "border-amber-500/50 bg-amber-500/10"
                    : "border-white/10 bg-white/5 hover:border-white/20"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-display font-semibold text-white/90">
                    {archetype.name}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-full border ${
                      archetype.defaultRiskCategory === "low"
                        ? "text-emerald-400 border-emerald-500/30 bg-emerald-500/10"
                        : archetype.defaultRiskCategory === "high"
                        ? "text-red-400 border-red-500/30 bg-red-500/10"
                        : "text-amber-400 border-amber-500/30 bg-amber-500/10"
                    }`}
                  >
                    {archetype.defaultRiskCategory}
                  </span>
                </div>
                <p className="text-sm text-white/40">{archetype.description}</p>
                <div className="flex items-center gap-4 mt-3 text-[11px] text-white/30">
                  <span>{archetype.hoursPerWeek}h/week</span>
                  <span>${Math.round(archetype.baseMu).toLocaleString()}/mo avg</span>
                </div>
              </button>
            ))}
          </div>

          {/* Chart and results */}
          <div className="lg:col-span-8">
            {/* Chart */}
            <div
              className={`bg-white/5 border border-white/10 rounded-2xl p-6 mb-6 transition-opacity duration-300 ${
                isAnimating ? "opacity-50" : "opacity-100"
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-white/50">
                  24-Month Income Projection
                </h3>
                <span className="text-[11px] text-white/30">
                  {selectedArchetype.name}
                </span>
              </div>

              <div className="h-64" style={{ minHeight: 256 }}>
                {mounted && chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={200}>
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="fanGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis
                        dataKey="month"
                        stroke="rgba(255,255,255,0.2)"
                        fontSize={10}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => (value % 6 === 0 ? `${value}mo` : "")}
                      />
                      <YAxis
                        stroke="rgba(255,255,255,0.2)"
                        fontSize={10}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                        domain={["dataMin - 500", "dataMax + 500"]}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(0,0,0,0.8)",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                        formatter={(value) => [`$${Number(value).toLocaleString()}`, ""]}
                        labelFormatter={(label) => `Month ${label}`}
                      />
                      {/* P10-P90 band */}
                      <Area
                        type="monotone"
                        dataKey="p90"
                        stroke="none"
                        fill="url(#fanGradient)"
                        fillOpacity={0.3}
                      />
                      <Area
                        type="monotone"
                        dataKey="p10"
                        stroke="none"
                        fill="#0a0a0f"
                        fillOpacity={1}
                      />
                      {/* P25-P75 band */}
                      <Area
                        type="monotone"
                        dataKey="p75"
                        stroke="none"
                        fill="#f59e0b"
                        fillOpacity={0.15}
                      />
                      <Area
                        type="monotone"
                        dataKey="p25"
                        stroke="none"
                        fill="#0a0a0f"
                        fillOpacity={1}
                      />
                      {/* Median line */}
                      <Area
                        type="monotone"
                        dataKey="p50"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        fill="none"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-white/30 text-sm">Loading simulation...</div>
                  </div>
                )}
              </div>
            </div>

            {/* Result cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-display font-bold text-white/90 mb-1">
                  {mounted && simulationData ? `$${Math.round(simulationData.summary.medianMonthlyIncome).toLocaleString()}` : "—"}
                </div>
                <div className="text-[11px] text-white/40">Median Income</div>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-display font-bold text-white/90 mb-1">
                  {mounted && simulationData ? `${((simulationData.summary.incomeVolatilityCv) * 100).toFixed(0)}%` : "—"}
                </div>
                <div className="text-[11px] text-white/40">Volatility (CV)</div>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-display font-bold text-white/90 mb-1">
                  {mounted && simulationData ? `${((simulationData.summary.probNegativeCashFlowAnyMonth) * 100).toFixed(0)}%` : "—"}
                </div>
                <div className="text-[11px] text-white/40">P(Negative CF)</div>
              </div>

              <div className={`rounded-xl p-4 border ${riskAssessment.bg}`}>
                <div className={`text-sm font-semibold ${riskAssessment.color} mb-1`}>
                  {riskAssessment.label}
                </div>
                <div className="text-[11px] text-white/40">vs FICO Score</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default PreviewSection;
