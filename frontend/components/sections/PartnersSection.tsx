"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const PARTNERS = [
  "Capital One",
  "University of Virginia",
  "JPMorgan Chase Institute",
  "Federal Reserve",
  "Uber",
  "Lyft",
  "DoorDash",
  "Instacart",
  "Gridwise",
  "Bureau of Labor Statistics",
];

export function PartnersSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const partnersRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      // Set initial states
      gsap.set([labelRef.current, partnersRef.current, ctaRef.current], {
        opacity: 0,
        y: 50,
      });

      // Create scroll-triggered animation
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 80%",
          toggleActions: "play none none none",
        },
      });

      tl.to(labelRef.current, {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: "power3.out",
      })
        .to(
          partnersRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 1,
            ease: "power3.out",
          },
          "-=0.5"
        )
        .to(
          ctaRef.current,
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "power3.out",
          },
          "-=0.4"
        );

      // Stagger partner names
      gsap.from(".partner-name", {
        scrollTrigger: {
          trigger: partnersRef.current,
          start: "top 85%",
        },
        opacity: 0,
        y: 20,
        stagger: 0.05,
        duration: 0.6,
        ease: "power3.out",
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      id="partners"
      ref={sectionRef}
      className="relative w-full pt-12 pb-16 px-8 md:px-16 lg:px-24 overflow-hidden"
    >
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-r from-amber-500/5 to-orange-500/5 blur-3xl" />
      </div>

      {/* Section label */}
      <div ref={labelRef} className="text-center mb-8">
        <span className="inline-flex items-center gap-3 text-[11px] font-medium tracking-[0.25em] uppercase text-white/30">
          <span className="w-8 h-px bg-white/20" />
          Data Sources & Inspiration
          <span className="w-8 h-px bg-white/20" />
        </span>
      </div>

      {/* Partners list - monopo style comma-separated */}
      <div
        ref={partnersRef}
        className="max-w-4xl mx-auto text-center mb-12"
      >
        <p className="font-display text-2xl md:text-3xl lg:text-4xl font-bold leading-relaxed tracking-tight">
          {PARTNERS.map((partner, index) => (
            <span key={partner}>
              <span className="partner-name inline-block text-white/80 hover:text-amber-400 transition-colors duration-300 cursor-default">
                {partner}
              </span>
              {index < PARTNERS.length - 1 && (
                <span className="text-white/30">, </span>
              )}
              {index === PARTNERS.length - 1 && (
                <span className="text-amber-400">.</span>
              )}
            </span>
          ))}
        </p>
      </div>

      {/* CTA */}
      <div ref={ctaRef} className="text-center">
        <a
          href="#"
          className="inline-flex items-center gap-3 text-sm text-white/40 hover:text-white/70 transition-colors duration-300 group"
        >
          <span className="w-6 h-px bg-white/20 group-hover:w-10 transition-all duration-300" />
          Explore our methodology
          <svg
            className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M13 7l5 5m0 0l-5 5m5-5H6"
            />
          </svg>
        </a>
      </div>
    </section>
  );
}

export default PartnersSection;
