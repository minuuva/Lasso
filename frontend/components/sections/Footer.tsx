"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const DATA_SOURCES = [
  { name: "JPMorgan Chase Institute", url: "https://www.jpmorganchase.com/institute" },
  { name: "Federal Reserve", url: "https://fred.stlouisfed.org" },
  { name: "Gridwise", url: "https://gridwise.io" },
  { name: "Bureau of Labor Statistics", url: "https://www.bls.gov" },
];

const SOCIAL_LINKS = [
  { name: "GitHub", url: "https://github.com", icon: "github" },
  { name: "LinkedIn", url: "https://linkedin.com", icon: "linkedin" },
  { name: "Twitter", url: "https://twitter.com", icon: "twitter" },
];

export function Footer() {
  const footerRef = useRef<HTMLElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!footerRef.current) return;

    const ctx = gsap.context(() => {
      gsap.set([ctaRef.current, contentRef.current], { opacity: 0, y: 50 });

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: footerRef.current,
          start: "top 80%",
          toggleActions: "play none none reverse",
        },
      });

      tl.to(ctaRef.current, {
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
      }).to(
        contentRef.current,
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: "power3.out",
        },
        "-=0.5"
      );
    }, footerRef);

    return () => ctx.revert();
  }, []);

  return (
    <footer ref={footerRef} className="relative w-full border-t border-white/[0.04] bg-gradient-to-b from-transparent to-black/40">
      {/* Keep in touch section - Enhanced CTA */}
      <div ref={ctaRef} className="px-8 md:px-16 lg:px-24 py-24 border-b border-white/[0.04]">
        <div className="max-w-5xl">
          {/* Section label - Tempo style */}
          <div className="section-label mb-10">
            <span>Contact</span>
          </div>

          {/* Large CTA headline */}
          <h2 className="font-display text-4xl md:text-5xl lg:text-[4rem] font-bold leading-[1.05] tracking-[-0.03em] text-white mb-10">
            Keep in touch
            <span className="inline-block ml-4 w-3 h-3 rounded-full bg-gradient-to-r from-amber-400 to-orange-500 animate-pulse" />
          </h2>

          {/* Email link - prominent with gradient hover */}
          <div className="mb-10">
            <span className="text-[10px] text-white/25 uppercase tracking-[0.15em] mb-3 block font-medium">
              Start a conversation
            </span>
            <a
              href="mailto:hello@lasso.io"
              className="group inline-flex items-center gap-3 font-display text-2xl md:text-3xl text-white/80 hover:text-gradient-accent transition-all duration-300"
            >
              <span>hello@lasso.io</span>
              <svg className="w-5 h-5 text-amber-400/50 group-hover:text-amber-400 group-hover:translate-x-1 transition-all" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </a>
          </div>

          {/* Social links - pill style */}
          <div className="flex flex-wrap items-center gap-3">
            {SOCIAL_LINKS.map((link) => (
              <a
                key={link.name}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded-full border border-white/10 text-sm text-white/40 hover:text-white/80 hover:border-white/20 hover:bg-white/[0.03] transition-all duration-300"
              >
                {link.name}
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* Main footer content */}
      <div ref={contentRef} className="px-8 md:px-16 lg:px-24 py-20">
        <div className="grid md:grid-cols-4 gap-12 mb-20">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 via-orange-500 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
                <span className="text-white font-bold text-base">L</span>
              </div>
              <span className="font-display font-bold text-2xl text-white">Lasso</span>
            </div>
            <p className="text-[15px] text-white/35 leading-relaxed max-w-md">
              Monte Carlo credit intelligence for the gig economy.
              Simulating thousands of financial futures to reveal what credit scores miss.
            </p>
          </div>

          {/* Data Sources */}
          <div>
            <h4 className="text-[10px] font-semibold uppercase tracking-[0.15em] text-white/40 mb-5">
              Data Sources
            </h4>
            <ul className="space-y-3">
              {DATA_SOURCES.map((source) => (
                <li key={source.name}>
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-white/35 hover:text-white/70 transition-colors duration-300"
                  >
                    {source.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="text-[10px] font-semibold uppercase tracking-[0.15em] text-white/40 mb-5">
              Built With
            </h4>
            <div className="flex flex-wrap gap-2">
              {["Next.js", "TypeScript", "Three.js", "GSAP", "Tailwind", "Claude AI"].map((tech) => (
                <span
                  key={tech}
                  className="text-[11px] px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06] text-white/35 hover:border-white/15 hover:text-white/50 transition-all duration-300"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-white/[0.04] flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r from-amber-500/10 to-orange-500/5 border border-amber-500/20">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span className="text-[11px] text-amber-400/80 font-medium">HooHacks 2026</span>
            </div>
            <span className="text-[11px] text-white/25">Capital One Finance Track</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[11px] text-white/20">© 2026 Lasso</span>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="w-9 h-9 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center text-white/30 hover:text-white/60 hover:border-white/15 transition-all duration-300"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
            </a>
          </div>
        </div>
      </div>

      {/* Large watermark - enhanced */}
      <div className="absolute bottom-0 left-0 right-0 overflow-hidden pointer-events-none">
        <div
          className="font-display font-bold text-[18vw] leading-none text-center -mb-[4vw]"
          style={{
            background: "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          LASSO
        </div>
      </div>
    </footer>
  );
}

export default Footer;
