"use client";

import React from "react";
import Image from "next/image";

const TEAM_MEMBERS = [
  {
    name: "Adam Carlson",
    role: "Physics & Math",
    year: "3rd Year @ UVA",
    company: "Susquehanna International Group",
    location: "Williamsburg, Virginia",
    image: "/team-adam.jpg",
  },
  {
    name: "Minu Choi",
    role: "Data Science",
    year: "3rd Year @ UVA",
    company: "Grid",
    location: "Seoul, South Korea",
    image: "/team-minu.jpg",
  },
  {
    name: "Otso Karali",
    role: "Math & Computer Science",
    year: "3rd Year @ UVA",
    company: "DRW Trading",
    location: "Helsinki, Finland",
    image: "/team-otso.png",
  },
];

export function TeamSection() {
  return (
    <section className="relative w-full pt-12 pb-16 px-8 md:px-16 lg:px-24 overflow-hidden">
      {/* Background gradient */}
      <div
        className="absolute right-0 top-1/2 -translate-y-1/2 w-[600px] h-[600px] pointer-events-none"
        style={{
          background: "radial-gradient(circle at center, rgba(255,159,64,0.05) 0%, transparent 60%)",
          filter: "blur(60px)",
        }}
      />

      {/* Section label */}
      <div className="section-label mb-10">
        <span>The Team</span>
      </div>

      {/* Headline */}
      <h2 className="font-display text-4xl md:text-5xl lg:text-[4rem] font-bold leading-[1.05] tracking-[-0.03em] mb-16">
        <span className="text-white">Built by</span>
        <br />
        <span className="text-gradient-accent">UVA students.</span>
      </h2>

      {/* Team cards */}
      <div className="grid md:grid-cols-3 gap-10 w-full">
        {TEAM_MEMBERS.map((member) => (
          <div
            key={member.name}
            className="p-8 rounded-2xl bg-white/[0.04] border border-white/[0.1] hover:border-amber-500/30 hover:bg-white/[0.06] transition-all duration-300"
          >
            {/* Photo */}
            <div className="relative w-40 h-40 mx-auto mb-8 rounded-2xl overflow-hidden border-2 border-white/[0.1]">
              <Image
                src={member.image}
                alt={member.name}
                fill
                className="object-cover"
              />
            </div>

            {/* Name */}
            <h3 className="text-2xl font-display font-semibold text-white text-center mb-2">
              {member.name}
            </h3>

            {/* Role & Year */}
            <p className="text-base text-white/70 text-center mb-6">
              {member.role} &middot; {member.year}
            </p>

            {/* Company & Location */}
            <div className="space-y-3">
              <div className="flex items-center justify-center gap-2 text-sm text-white/60">
                <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
                <span>{member.company}</span>
              </div>
              <div className="flex items-center justify-center gap-2 text-sm text-white/60">
                <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span>{member.location}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default TeamSection;
