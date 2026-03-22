"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { CountUp } from "@/components/viz/CountUp";
import { IncomeChart } from "@/components/viz/IncomeChart";

gsap.registerPlugin(ScrollTrigger);

export function ProblemSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const sublineRef = useRef<HTMLParagraphElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      // Set initial states
      gsap.set([headlineRef.current, sublineRef.current, chartRef.current, statsRef.current], {
        opacity: 0,
        y: 80,
      });

      // Create scroll-triggered timeline
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 60%",
          end: "center center",
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
          sublineRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "power3.out",
          },
          "-=0.6"
        )
        .to(
          chartRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 1,
            ease: "power3.out",
          },
          "-=0.4"
        )
        .to(
          statsRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "power3.out",
          },
          "-=0.6"
        );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full py-24 px-8 md:px-16 lg:px-24 flex flex-col justify-center overflow-hidden"
    >
      {/* Background gradient accent */}
      <div
        className="absolute -right-40 top-1/4 w-[600px] h-[600px] pointer-events-none"
        style={{
          background: "radial-gradient(circle at center, rgba(248,113,113,0.08) 0%, transparent 60%)",
          filter: "blur(60px)",
        }}
      />

      {/* Section label - Tempo style */}
      <div className="section-label mb-10">
        <span>The Problem</span>
      </div>

      {/* Main headline - Stripe Sessions gradient text */}
      <h2
        ref={headlineRef}
        className="font-display text-4xl md:text-5xl lg:text-[4rem] font-bold leading-[1.05] tracking-[-0.03em] mb-8"
      >
        <span className="text-white">FICO sees your past.</span>
        <br />
        <span className="text-gradient-fade">Not your future.</span>
      </h2>

      {/* Subline */}
      <p
        ref={sublineRef}
        className="max-w-xl text-xl text-white/40 leading-relaxed mb-20 font-light"
      >
        Traditional credit models assume stable paychecks.{" "}
        <span className="text-white/70 font-normal">
          Gig workers don&apos;t have that luxury.
        </span>
      </p>

      {/* Income Chart - in a glass card */}
      <div ref={chartRef} className="max-w-4xl mb-20">
        <div className="glass-card-premium p-8">
          <IncomeChart />
        </div>
      </div>

      {/* Stats row - Tempo numbered style */}
      <div
        ref={statsRef}
        className="grid grid-cols-1 md:grid-cols-3 gap-0 max-w-4xl"
      >
        {/* Stat 1: Volatility */}
        <div className="numbered-card" data-number="01">
          <div className="text-5xl md:text-6xl font-display font-bold text-gradient-accent mb-3">
            <CountUp end={36} suffix="%" />
          </div>
          <div className="text-sm text-white/40 leading-relaxed">
            Month-to-month income volatility for gig workers
          </div>
        </div>

        {/* Stat 2: Workers affected */}
        <div className="numbered-card" data-number="02">
          <div className="text-5xl md:text-6xl font-display font-bold text-white mb-3">
            <CountUp end={59} suffix="M" />
          </div>
          <div className="text-sm text-white/40 leading-relaxed">
            Americans participated in gig work in 2024
          </div>
        </div>

        {/* Stat 3: FICO blind */}
        <div className="numbered-card" data-number="03">
          <div className="text-5xl md:text-6xl font-display font-bold text-white/50 mb-3">
            <CountUp end={0} suffix="%" />
          </div>
          <div className="text-sm text-white/40 leading-relaxed">
            of income volatility captured by FICO scores
          </div>
        </div>
      </div>
    </section>
  );
}

export default ProblemSection;
