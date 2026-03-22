"use client";

import React, { useEffect, useRef, useMemo } from "react";
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

// Generate a single Monte Carlo path
function generatePath(seed: number): number[] {
  const points: number[] = [0];
  let value = 0;
  const random = seededRandom(seed * 1000);

  for (let i = 1; i <= 36; i++) {
    // Random walk with drift
    const drift = 0.002;
    const volatility = 0.08 + (seed % 10) * 0.01;
    const noise = (random() - 0.5) * 2 * volatility;
    value += drift + noise;
    points.push(value);
  }

  return points;
}

export function PathsAnimation({ className = "" }: { className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pathsRef = useRef<SVGPathElement[]>([]);

  const numPaths = 50; // Visual representation (actual sim uses 5000)
  const width = 600;
  const height = 400;

  // Generate paths data
  const pathsData = useMemo(() => {
    return Array(numPaths)
      .fill(0)
      .map((_, i) => generatePath(i));
  }, []);

  // Convert data to SVG path
  const dataToPath = (data: number[]): string => {
    const xScale = width / (data.length - 1);
    const yCenter = height / 2;
    const yScale = height * 0.4;

    return data.reduce((path, value, i) => {
      const x = i * xScale;
      const y = yCenter - value * yScale;
      return path + (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }, "");
  };

  useEffect(() => {
    if (!containerRef.current) return;

    const paths = pathsRef.current;

    // Set initial state
    paths.forEach((path) => {
      if (!path) return;
      const length = path.getTotalLength();
      gsap.set(path, {
        strokeDasharray: length,
        strokeDashoffset: length,
        opacity: 0,
      });
    });

    // Create scroll animation
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top 60%",
        end: "center center",
        scrub: 1,
      },
    });

    // Animate paths in sequence
    paths.forEach((path, i) => {
      if (!path) return;
      tl.to(
        path,
        {
          strokeDashoffset: 0,
          opacity: 0.6 - (i / numPaths) * 0.4,
          duration: 1,
          ease: "none",
        },
        i * 0.02
      );
    });

    return () => {
      tl.kill();
    };
  }, []);

  // Color based on final value (green = up, red = down)
  const getPathColor = (data: number[]): string => {
    const finalValue = data[data.length - 1];
    if (finalValue > 0.3) return "#22c55e"; // green
    if (finalValue < -0.3) return "#ef4444"; // red
    return "#f59e0b"; // amber
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Background grid */}
        <defs>
          <pattern
            id="grid"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="rgba(255,255,255,0.03)"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Center line (starting point) */}
        <line
          x1="0"
          y1={height / 2}
          x2="30"
          y2={height / 2}
          stroke="#f59e0b"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Monte Carlo paths */}
        {pathsData.map((data, i) => (
          <path
            key={i}
            ref={(el) => {
              if (el) pathsRef.current[i] = el;
            }}
            d={dataToPath(data)}
            fill="none"
            stroke={getPathColor(data)}
            strokeWidth="1.5"
            strokeLinecap="round"
            opacity="0"
          />
        ))}

        {/* Labels */}
        <text x="10" y="30" className="fill-white/40 text-[12px]">
          Start
        </text>
        <text x={width - 60} y="30" className="fill-white/40 text-[12px]">
          36 months
        </text>

        {/* Fan indicator */}
        <text
          x={width / 2}
          y={height - 20}
          textAnchor="middle"
          className="fill-white/30 text-[11px]"
        >
          5,000 simulated paths
        </text>
      </svg>

      {/* Percentile labels */}
      <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col items-end gap-1">
        <span className="text-[10px] text-green-400/60">p90</span>
        <span className="text-[10px] text-amber-400/80">p50</span>
        <span className="text-[10px] text-red-400/60">p10</span>
      </div>
    </div>
  );
}

export default PathsAnimation;
