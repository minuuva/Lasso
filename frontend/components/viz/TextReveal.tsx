"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

interface TextRevealProps {
  children: string;
  className?: string;
  revealedColor?: string;
  unrevealedColor?: string;
  staggerDelay?: number;
}

export function TextReveal({
  children,
  className = "",
  revealedColor = "text-white",
  unrevealedColor = "text-white/20",
  staggerDelay = 0.03,
}: TextRevealProps) {
  const containerRef = useRef<HTMLSpanElement>(null);
  const wordsRef = useRef<HTMLSpanElement[]>([]);

  const words = children.split(" ");

  useEffect(() => {
    if (!containerRef.current || wordsRef.current.length === 0) return;

    const ctx = gsap.context(() => {
      // Set initial state
      gsap.set(wordsRef.current, {
        className: `inline-block ${unrevealedColor} transition-colors duration-500`
      });

      // Animate on scroll
      ScrollTrigger.create({
        trigger: containerRef.current,
        start: "top 80%",
        end: "bottom 40%",
        scrub: 0.5,
        onUpdate: (self) => {
          const progress = self.progress;
          const totalWords = wordsRef.current.length;
          const currentWordIndex = Math.floor(progress * totalWords);

          wordsRef.current.forEach((word, index) => {
            if (index <= currentWordIndex) {
              word.className = `inline-block ${revealedColor} transition-colors duration-300`;
            } else {
              word.className = `inline-block ${unrevealedColor} transition-colors duration-300`;
            }
          });
        },
      });
    }, containerRef);

    return () => ctx.revert();
  }, [revealedColor, unrevealedColor]);

  return (
    <span ref={containerRef} className={className}>
      {words.map((word, index) => (
        <span key={index}>
          <span
            ref={(el) => {
              if (el) wordsRef.current[index] = el;
            }}
            className={`inline-block ${unrevealedColor}`}
          >
            {word}
          </span>
          {index < words.length - 1 && " "}
        </span>
      ))}
    </span>
  );
}

export default TextReveal;
