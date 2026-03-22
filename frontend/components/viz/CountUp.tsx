"use client";

import React, { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

interface CountUpProps {
  end: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  className?: string;
}

export function CountUp({
  end,
  duration = 2,
  suffix = "",
  prefix = "",
  decimals = 0,
  className = "",
}: CountUpProps) {
  const [count, setCount] = useState(0);
  const elementRef = useRef<HTMLSpanElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (!elementRef.current) return;

    const trigger = ScrollTrigger.create({
      trigger: elementRef.current,
      start: "top 80%",
      onEnter: () => {
        if (hasAnimated.current) return;
        hasAnimated.current = true;

        gsap.to(
          { value: 0 },
          {
            value: end,
            duration,
            ease: "power2.out",
            onUpdate: function () {
              setCount(this.targets()[0].value);
            },
          }
        );
      },
    });

    return () => trigger.kill();
  }, [end, duration]);

  const formattedCount = decimals > 0
    ? count.toFixed(decimals)
    : Math.round(count).toLocaleString();

  return (
    <span ref={elementRef} className={className}>
      {prefix}{formattedCount}{suffix}
    </span>
  );
}

export default CountUp;
