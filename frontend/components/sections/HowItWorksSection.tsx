"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { PathsAnimation } from "@/components/viz/PathsAnimation";

gsap.registerPlugin(ScrollTrigger);

const STEPS = [
  {
    number: "01",
    title: "Model the Worker",
    description: "Platform, hours, expenses, savings. We capture the full picture of gig life.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
      </svg>
    ),
  },
  {
    number: "02",
    title: "Run the Paths",
    description: "5,000 simulations accounting for seasonality, macro shocks, and life events.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
      </svg>
    ),
  },
  {
    number: "03",
    title: "Assess the Risk",
    description: "Compare our probability distribution against what FICO would predict.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
      </svg>
    ),
  },
];

export function HowItWorksSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const stepsRef = useRef<HTMLDivElement>(null);
  const vizRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      // Set initial states
      gsap.set([headlineRef.current, stepsRef.current, vizRef.current], {
        opacity: 0,
        y: 60,
      });

      // Main timeline
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 60%",
          toggleActions: "play none none reverse",
        },
      });

      tl.to(headlineRef.current, {
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
      })
        .to(
          vizRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 1,
            ease: "power3.out",
          },
          "-=0.5"
        )
        .to(
          stepsRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "power3.out",
          },
          "-=0.6"
        );

      // Stagger step cards
      gsap.from(".step-card", {
        scrollTrigger: {
          trigger: stepsRef.current,
          start: "top 70%",
        },
        opacity: 0,
        y: 40,
        stagger: 0.15,
        duration: 0.8,
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
      {/* Background gradient */}
      <div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] pointer-events-none"
        style={{
          background: "radial-gradient(circle at center, rgba(255,159,64,0.04) 0%, transparent 50%)",
        }}
      />

      {/* Section label - Tempo style */}
      <div className="section-label mb-10">
        <span>How It Works</span>
      </div>

      {/* Main headline */}
      <h2
        ref={headlineRef}
        className="font-display text-4xl md:text-5xl lg:text-[4rem] font-bold leading-[1.05] tracking-[-0.03em] mb-20"
      >
        <span className="text-white">We simulate</span>
        <br />
        <span className="text-gradient-accent">5,000 possible futures.</span>
      </h2>

      {/* Two-column layout */}
      <div className="grid lg:grid-cols-2 gap-20 items-start max-w-7xl">
        {/* Paths visualization - in glass card */}
        <div ref={vizRef} className="order-2 lg:order-1">
          <div className="glass-card-premium p-6 md:p-8">
            <PathsAnimation className="w-full" />
          </div>
          <p className="text-sm text-white/30 mt-6 text-center font-light">
            Each path represents a possible financial trajectory based on real volatility data
          </p>
        </div>

        {/* Steps - Tempo numbered cards */}
        <div ref={stepsRef} className="order-1 lg:order-2">
          {STEPS.map((step) => (
            <div
              key={step.number}
              className="step-card numbered-card group"
              data-number={step.number}
            >
              {/* Icon */}
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20 flex items-center justify-center mb-5 group-hover:border-amber-500/40 transition-colors">
                <div className="text-amber-400/80 group-hover:text-amber-400 transition-colors">
                  {step.icon}
                </div>
              </div>

              {/* Title */}
              <h3 className="text-xl font-display font-semibold text-white mb-3">
                {step.title}
              </h3>

              {/* Description */}
              <p className="text-white/40 leading-relaxed text-[15px]">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default HowItWorksSection;
