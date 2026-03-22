"use client";

import React from "react";

interface ScrollIndicatorProps {
  className?: string;
}

export function ScrollIndicator({ className = "" }: ScrollIndicatorProps) {
  return (
    <div className={`flex flex-col items-center gap-3 ${className}`}>
      <span className="text-[9px] text-white/40 uppercase tracking-[0.2em] font-medium">
        Scroll
      </span>
      <div className="relative w-px h-16">
        <div className="absolute inset-0 bg-gradient-to-b from-white/30 to-transparent" />
        <div className="absolute top-0 w-px h-8 bg-white/50 animate-scroll-line" />
      </div>
    </div>
  );
}

export default ScrollIndicator;
