"use client";

// Force cache invalidation
import React from "react";
import { HeroSection } from "@/components/hero";
import {
  ProblemSection,
  HowItWorksSection,
  PreviewSection,
  ManifestoSection,
  PartnersSection,
  AISection,
  Footer,
} from "@/components/sections";

export default function Home() {
  return (
    <main className="bg-[#0a0a0f]">
      {/* Hero */}
      <HeroSection />

      {/* Section 1: The Problem */}
      <ProblemSection />

      {/* Section 2: How It Works */}
      <HowItWorksSection />

      {/* Section 3: Manifesto - monopo style text reveal */}
      <ManifestoSection />

      {/* Section 4: AI Scenario Analysis */}
      <AISection />

      {/* Section 5: Live Preview */}
      <PreviewSection />

      {/* Section 6: Partners & Data Sources */}
      <PartnersSection />

      {/* Footer */}
      <Footer />
    </main>
  );
}
