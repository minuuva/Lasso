"use client";

import React, { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// Seeded random number generator for consistent SSR/client values
function seededRandom(seed: number): () => number {
  return () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
}

// Generate realistic gig income data (volatile)
function generateGigIncome(months: number, seed: number): number[] {
  const base = 3200;
  const data: number[] = [];
  const random = seededRandom(seed);

  for (let i = 0; i < months; i++) {
    // High volatility with seasonal patterns
    const seasonal = Math.sin((i / 12) * Math.PI * 2) * 400;
    const noise = (random() - 0.5) * 1200;
    const spike = random() > 0.85 ? (random() * 800) : 0;
    const dip = random() > 0.9 ? -(random() * 600) : 0;

    data.push(Math.max(1800, base + seasonal + noise + spike + dip));
  }

  return data;
}

// Generate stable W-2 income data
function generateW2Income(months: number, seed: number): number[] {
  const base = 3500;
  const random = seededRandom(seed);
  return Array(months).fill(0).map((_, i) => {
    // Very slight variation, occasional small raise
    const raise = Math.floor(i / 12) * 100;
    const variation = (random() - 0.5) * 50;
    return base + raise + variation;
  });
}

export function IncomeChart({ className = "" }: { className?: string }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gigPathRef = useRef<SVGPathElement>(null);
  const w2PathRef = useRef<SVGPathElement>(null);
  const [mounted, setMounted] = useState(false);

  const months = 24;
  // Use fixed seeds for consistent SSR/client rendering
  const gigData = useRef(generateGigIncome(months, 12345));
  const w2Data = useRef(generateW2Income(months, 67890));

  useEffect(() => {
    setMounted(true);
  }, []);

  // SVG dimensions
  const width = 800;
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Scale functions
  const xScale = (i: number) => padding.left + (i / (months - 1)) * chartWidth;
  const yScale = (value: number) => {
    const minY = 1500;
    const maxY = 5000;
    return padding.top + chartHeight - ((value - minY) / (maxY - minY)) * chartHeight;
  };

  // Generate path string
  const generatePath = (data: number[]): string => {
    return data.reduce((path, value, i) => {
      const x = xScale(i);
      const y = yScale(value);
      return path + (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }, "");
  };

  useEffect(() => {
    if (!svgRef.current || !gigPathRef.current || !w2PathRef.current) return;

    const gigPath = gigPathRef.current;
    const w2Path = w2PathRef.current;

    // Get path lengths
    const gigLength = gigPath.getTotalLength();
    const w2Length = w2Path.getTotalLength();

    // Set initial state (hidden)
    gsap.set(gigPath, {
      strokeDasharray: gigLength,
      strokeDashoffset: gigLength,
    });
    gsap.set(w2Path, {
      strokeDasharray: w2Length,
      strokeDashoffset: w2Length,
    });

    // Animate on scroll
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: svgRef.current,
        start: "top 70%",
        end: "center center",
        scrub: 1,
      },
    });

    tl.to(w2Path, {
      strokeDashoffset: 0,
      duration: 1,
      ease: "none",
    }).to(
      gigPath,
      {
        strokeDashoffset: 0,
        duration: 1,
        ease: "none",
      },
      0.2
    );

    return () => {
      tl.kill();
    };
  }, []);

  return (
    <div className={`relative ${className}`}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        {[2000, 2500, 3000, 3500, 4000, 4500].map((value) => (
          <g key={value}>
            <line
              x1={padding.left}
              y1={yScale(value)}
              x2={width - padding.right}
              y2={yScale(value)}
              stroke="rgba(255,255,255,0.05)"
              strokeDasharray="4 4"
            />
            <text
              x={padding.left - 10}
              y={yScale(value)}
              textAnchor="end"
              alignmentBaseline="middle"
              className="fill-white/30 text-[10px]"
            >
              ${(value / 1000).toFixed(1)}k
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {[0, 6, 12, 18, 23].map((i) => (
          <text
            key={i}
            x={xScale(i)}
            y={height - 10}
            textAnchor="middle"
            className="fill-white/30 text-[10px]"
          >
            {i === 0 ? "Now" : `+${i}mo`}
          </text>
        ))}

        {/* W-2 Income line (stable) */}
        <path
          ref={w2PathRef}
          d={generatePath(w2Data.current)}
          fill="none"
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* Gig Income line (volatile) */}
        <path
          ref={gigPathRef}
          d={generatePath(gigData.current)}
          fill="none"
          stroke="url(#gigGradient)"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Gradient definition */}
        <defs>
          <linearGradient id="gigGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="50%" stopColor="#f97316" />
            <stop offset="100%" stopColor="#ea580c" />
          </linearGradient>
        </defs>
      </svg>

      {/* Legend */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex items-center gap-8 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-0.5 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full" />
          <span className="text-xs text-white/60">Gig Income</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-0.5 bg-white/30 rounded-full" />
          <span className="text-xs text-white/40">W-2 Income</span>
        </div>
      </div>
    </div>
  );
}

export default IncomeChart;
