"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { AIChat } from "@/components/chat";

gsap.registerPlugin(ScrollTrigger);

export function AISection() {
  const sectionRef = useRef<HTMLElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set(contentRef.current, { opacity: 0, y: 60 });

      gsap.to(contentRef.current, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 60%",
          toggleActions: "play none none reverse",
        },
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full py-24 px-8 md:px-16 lg:px-24 overflow-hidden"
    >
      {/* Glowing platform effect - Reflect style with smooth transitions */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {/* Top fade - smooth transition from previous section */}
        <div
          className="absolute top-0 left-0 right-0 h-32"
          style={{
            background: "linear-gradient(to bottom, rgba(10,10,15,0.8) 0%, transparent 100%)",
          }}
        />
        {/* Main glow - extended and more gradual fade */}
        <div
          className="absolute left-1/2 -translate-x-1/2 bottom-0 w-[1000px] h-[500px]"
          style={{
            background: `
              radial-gradient(ellipse 100% 80% at 50% 100%,
                rgba(139,92,246,0.12) 0%,
                rgba(168,85,247,0.06) 40%,
                rgba(139,92,246,0.02) 70%,
                transparent 100%
              )
            `,
            filter: "blur(60px)",
          }}
        />
        {/* Light beam from bottom */}
        <div
          className="absolute left-1/2 -translate-x-1/2 bottom-0 w-[3px] h-[50%]"
          style={{
            background: "linear-gradient(to top, rgba(139,92,246,0.3), rgba(255,159,64,0.15), transparent)",
            filter: "blur(4px)",
          }}
        />
        {/* Bottom fade - smooth transition to next section */}
        <div
          className="absolute bottom-0 left-0 right-0 h-40"
          style={{
            background: "linear-gradient(to top, rgba(10,10,15,1) 0%, rgba(10,10,15,0.5) 50%, transparent 100%)",
          }}
        />
      </div>

      <div ref={contentRef}>
        {/* Pill badge */}
        <div className="mb-10">
          <span className="pill-badge">
            <svg className="w-4 h-4 text-purple-400" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0L14.59 8.41L23 11L14.59 13.59L12 22L9.41 13.59L1 11L9.41 8.41L12 0Z" />
            </svg>
            <span>AI-Powered Analysis</span>
          </span>
        </div>

        {/* Headline */}
        <h2 className="font-display text-4xl md:text-5xl lg:text-[4rem] font-bold leading-[1.05] tracking-[-0.03em] mb-8">
          <span className="text-white">Ask</span>{" "}
          <span className="text-gradient-vivid">&quot;What if?&quot;</span>
        </h2>

        <p className="max-w-xl text-xl text-white/40 leading-relaxed mb-16 font-light">
          Describe any financial scenario in plain English. Our AI interprets your question
          and runs thousands of Monte Carlo simulations to show you the real impact.
        </p>

        {/* Two column layout */}
        <div className="grid lg:grid-cols-2 gap-12 items-start max-w-7xl">
          {/* Chat interface - in glass card with glow */}
          <div className="relative">
            <div className="absolute -inset-4 bg-gradient-to-r from-purple-500/10 via-amber-500/5 to-purple-500/10 rounded-3xl blur-2xl opacity-50" />
            <div className="relative glass-card-premium h-[600px]">
              <AIChat />
            </div>
          </div>

          {/* Info cards - redesigned */}
          <div className="space-y-6">
            {/* How It Works */}
            <div className="glass-card-premium p-8">
              <h3 className="font-display font-semibold text-white text-lg mb-6">
                How It Works
              </h3>
              <ol className="space-y-5">
                {[
                  "You describe a scenario: \"What if gas prices spike 30%?\"",
                  "AI converts this to simulation parameters",
                  "Monte Carlo engine runs 5,000 simulations",
                  "You see the updated risk metrics instantly",
                ].map((text, i) => (
                  <li key={i} className="flex gap-4 items-start">
                    <span className="flex-shrink-0 w-7 h-7 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 border border-amber-500/20 text-amber-400 text-xs font-mono flex items-center justify-center">
                      {i + 1}
                    </span>
                    <span className="text-[15px] text-white/50 leading-relaxed pt-0.5">
                      {text}
                    </span>
                  </li>
                ))}
              </ol>
            </div>

            {/* Example Scenarios */}
            <div className="glass-card-premium p-8">
              <h3 className="font-display font-semibold text-white text-lg mb-6">
                Example Scenarios
              </h3>
              <div className="grid gap-4">
                {[
                  { label: "Macro shocks", examples: "Recession, gas spikes, rate cuts" },
                  { label: "Life events", examples: "Injury, car breakdown, emergency" },
                  { label: "Market changes", examples: "Competitors, seasonality, regulations" },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-2 flex-shrink-0" />
                    <div>
                      <span className="text-white/70 font-medium text-sm">{item.label}</span>
                      <span className="text-white/40 text-sm"> — {item.examples}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Powered by Claude */}
            <div className="border-gradient-animated p-6">
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-amber-500/10 border border-purple-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="font-display font-semibold text-white">
                    Powered by Claude AI
                  </h3>
                </div>
                <p className="text-sm text-white/40 leading-relaxed">
                  Natural language understanding combined with rigorous Monte
                  Carlo simulation for accurate risk assessment.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default AISection;
