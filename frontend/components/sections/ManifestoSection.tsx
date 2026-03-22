"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function ManifestoSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const titleRef = useRef<HTMLDivElement>(null);
  const wordsRef = useRef<(HTMLSpanElement | null)[]>([]);
  const valuesRef = useRef<HTMLDivElement>(null);

  const manifesto = "We simulate thousands of futures to understand the present. We see what credit scores miss. We believe gig workers deserve better than a number.";
  const words = manifesto.split(" ");

  useEffect(() => {
    if (!sectionRef.current) return;

    const ctx = gsap.context(() => {
      // Title animation
      gsap.set(titleRef.current, { opacity: 0, y: 40 });

      gsap.to(titleRef.current, {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top 70%",
        },
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
      });

      // Word-by-word reveal on scroll
      const validWords = wordsRef.current.filter(Boolean) as HTMLSpanElement[];

      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: "top 70%",
        end: "top 10%",
        scrub: 0.3,
        onUpdate: (self) => {
          const progress = self.progress;
          const totalWords = validWords.length;
          // Ensure all words are revealed by progress 1.0
          const currentWordIndex = Math.floor(progress * (totalWords + 1));

          validWords.forEach((word, index) => {
            if (index <= currentWordIndex) {
              gsap.to(word, {
                color: "rgba(255, 255, 255, 0.95)",
                duration: 0.3,
              });
            } else {
              gsap.to(word, {
                color: "rgba(255, 255, 255, 0.15)",
                duration: 0.3,
              });
            }
          });
        },
      });

      // Values section animation
      gsap.set(valuesRef.current, { opacity: 0, y: 60 });

      gsap.to(valuesRef.current, {
        scrollTrigger: {
          trigger: valuesRef.current,
          start: "top 80%",
        },
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
      });

      // Stagger value cards
      gsap.from(".value-card", {
        scrollTrigger: {
          trigger: valuesRef.current,
          start: "top 70%",
        },
        opacity: 0,
        x: -30,
        stagger: 0.2,
        duration: 0.8,
        ease: "power3.out",
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  const values = [
    {
      number: "01",
      title: "SIMULATE",
      description:
        "Monte Carlo paths explore thousands of income scenarios—capturing seasonality, market shocks, and the chaos of gig life.",
    },
    {
      number: "02",
      title: "UNDERSTAND",
      description:
        "Transform raw volatility into actionable intelligence. Not a score—a probability distribution of financial futures.",
    },
    {
      number: "03",
      title: "EMPOWER",
      description:
        "Give lenders the tools to say yes to good borrowers that traditional models would reject. Fairness through simulation.",
    },
  ];

  return (
    <section
      ref={sectionRef}
      className="relative w-full py-24 px-8 md:px-16 lg:px-24 overflow-hidden"
    >
      {/* Background gradients - Stripe Sessions style */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute left-0 top-0 w-full h-full"
          style={{
            background: `
              radial-gradient(ellipse 50% 50% at 20% 30%, rgba(139,92,246,0.06) 0%, transparent 50%),
              radial-gradient(ellipse 60% 60% at 80% 70%, rgba(255,159,64,0.05) 0%, transparent 50%)
            `,
          }}
        />
      </div>

      {/* Section label - Tempo style */}
      <div ref={titleRef} className="section-label mb-16">
        <span>Manifesto</span>
      </div>

      {/* Main manifesto text - word by word reveal with gradient */}
      <div className="max-w-5xl mb-32">
        <p className="font-display text-3xl md:text-4xl lg:text-5xl xl:text-[3.5rem] font-semibold leading-[1.25] tracking-[-0.02em]">
          {words.map((word, index) => (
            <span key={index}>
              <span
                ref={(el) => {
                  wordsRef.current[index] = el;
                }}
                className="inline-block transition-colors duration-500"
                style={{ color: "rgba(255, 255, 255, 0.12)" }}
              >
                {word}
              </span>
              {index < words.length - 1 && " "}
            </span>
          ))}
        </p>
      </div>

      {/* Value pillars - Enhanced design */}
      <div ref={valuesRef} className="grid md:grid-cols-3 gap-0">
        {values.map((value, idx) => (
          <div
            key={value.number}
            className="value-card numbered-card group"
            data-number={value.number}
          >
            {/* Title */}
            <h3 className="font-display text-xl font-bold tracking-[0.08em] text-white/80 group-hover:text-gradient-accent transition-all duration-500 mb-5">
              {value.title}
            </h3>

            {/* Description */}
            <p className="text-[15px] text-white/35 leading-relaxed group-hover:text-white/50 transition-colors duration-500">
              {value.description}
            </p>

            {/* Arrow link */}
            <div className="mt-6 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
              <span className="inline-flex items-center gap-2 text-sm text-amber-400/70 font-medium">
                Learn more
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Decorative light beam */}
      <div
        className="absolute right-[20%] top-0 w-[2px] h-full pointer-events-none opacity-20"
        style={{
          background: "linear-gradient(to bottom, transparent 0%, rgba(255,159,64,0.3) 30%, rgba(139,92,246,0.2) 70%, transparent 100%)",
          filter: "blur(3px)",
        }}
      />
    </section>
  );
}

export default ManifestoSection;
