"use client";

import React, { useEffect, useRef, useMemo, Suspense, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import gsap from "gsap";
import Link from "next/link";

// Metro coordinates (lat, lon) for the 6 specified metros
const METROS = [
  { name: "New York", lat: 40.7128, lon: -74.006 },
  { name: "Los Angeles", lat: 34.0522, lon: -118.2437 },
  { name: "Chicago", lat: 41.8781, lon: -87.6298 },
  { name: "San Francisco", lat: 37.7749, lon: -122.4194 },
  { name: "Washington DC", lat: 38.9072, lon: -77.0369 },
  { name: "Richmond", lat: 37.5407, lon: -77.436 },
];

// Convert lat/lon to 3D position on sphere
function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

// Enhanced atmosphere shader - Vercel/Stripe globe style
const atmosphereVertexShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const atmosphereFragmentShader = `
  uniform float uTime;
  varying vec3 vNormal;
  varying vec3 vPosition;

  void main() {
    float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
    float pulse = 0.92 + 0.08 * sin(uTime * 0.3);

    // More sophisticated color gradient
    vec3 innerColor = vec3(1.0, 0.65, 0.3);  // Warm amber
    vec3 outerColor = vec3(1.0, 0.35, 0.15); // Deep orange
    vec3 glowColor = mix(innerColor, outerColor, intensity * 0.6);

    // Add subtle blue tint at edges
    vec3 edgeColor = vec3(0.4, 0.5, 0.9);
    glowColor = mix(glowColor, edgeColor, pow(intensity, 3.0) * 0.15);

    gl_FragColor = vec4(glowColor * intensity * pulse, intensity * 0.7);
  }
`;

// Enhanced Particle Globe with better visuals
function ParticleGlobe() {
  const pointsRef = useRef<THREE.Points>(null);
  const groupRef = useRef<THREE.Group>(null);
  const hotspotRefs = useRef<THREE.Mesh[]>([]);
  const atmosphereRef = useRef<THREE.Mesh>(null);
  const innerGlobeRef = useRef<THREE.Mesh>(null);

  const particleCount = 6000;
  const radius = 2.0;

  // Generate particles with Fibonacci sphere distribution for even coverage
  const { positions, colors, sizes } = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    const siz = new Float32Array(particleCount);

    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    for (let i = 0; i < particleCount; i++) {
      const theta = (2 * Math.PI * i) / goldenRatio;
      const phi = Math.acos(1 - (2 * (i + 0.5)) / particleCount);

      const x = radius * Math.cos(theta) * Math.sin(phi);
      const y = radius * Math.sin(theta) * Math.sin(phi);
      const z = radius * Math.cos(phi);

      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;

      // Sophisticated color gradient with depth variation
      const depth = (z / radius + 1) * 0.5; // 0 to 1 based on z position
      const brightness = 0.4 + depth * 0.6;

      // Mix between warm amber and subtle white highlights
      const highlight = Math.random() > 0.95 ? 1.2 : 1.0;
      col[i * 3] = Math.min(1.0, 1.0 * brightness * highlight);
      col[i * 3 + 1] = Math.min(1.0, 0.6 * brightness * highlight);
      col[i * 3 + 2] = Math.min(1.0, 0.2 * brightness * highlight);

      // Vary particle sizes for depth
      siz[i] = 0.015 + Math.random() * 0.025;
    }

    return { positions: pos, colors: col, sizes: siz };
  }, []);

  // Metro positions
  const metroPositions = useMemo(() => {
    return METROS.map((m) => latLonToVector3(m.lat, m.lon, radius));
  }, []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;

    // Smooth, slow rotation
    if (groupRef.current) {
      groupRef.current.rotation.y = t * 0.04;
      groupRef.current.rotation.x = Math.sin(t * 0.015) * 0.03;
    }

    // Animate atmosphere shader
    if (atmosphereRef.current) {
      const material = atmosphereRef.current.material as THREE.ShaderMaterial & { uTime: number };
      if (material.uniforms) {
        material.uniforms.uTime.value = t;
      }
    }

    // Subtle inner globe animation
    if (innerGlobeRef.current) {
      const material = innerGlobeRef.current.material as THREE.MeshBasicMaterial;
      material.opacity = 0.03 + Math.sin(t * 0.5) * 0.01;
    }

    // Pulse hotspots with staggered timing
    hotspotRefs.current.forEach((mesh, i) => {
      if (mesh) {
        const scale = 1 + Math.sin(t * 2 + i * 1.2) * 0.35;
        mesh.scale.setScalar(scale * 0.055);
      }
    });
  });

  return (
    <group ref={groupRef} rotation={[0.15, -0.8, 0.05]}>
      {/* Outer atmosphere glow - enhanced layers */}
      <mesh ref={atmosphereRef} scale={[1.28, 1.28, 1.28]}>
        <sphereGeometry args={[radius, 64, 64]} />
        <shaderMaterial
          vertexShader={atmosphereVertexShader}
          fragmentShader={atmosphereFragmentShader}
          uniforms={{ uTime: { value: 0 } }}
          transparent
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Secondary atmosphere layer */}
      <mesh scale={[1.18, 1.18, 1.18]}>
        <sphereGeometry args={[radius, 48, 48]} />
        <meshBasicMaterial
          color="#ff6b35"
          transparent
          opacity={0.06}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Core glow layer */}
      <mesh scale={[1.08, 1.08, 1.08]}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshBasicMaterial
          color="#ff9f40"
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Inner solid sphere for depth */}
      <mesh ref={innerGlobeRef} scale={[0.98, 0.98, 0.98]}>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshBasicMaterial
          color="#1a1a2e"
          transparent
          opacity={0.04}
        />
      </mesh>

      {/* Main particle sphere */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.028}
          vertexColors
          transparent
          opacity={0.92}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>

      {/* Metro hotspots - refined pulsing dots */}
      {metroPositions.map((pos, i) => (
        <group key={i} position={pos}>
          {/* Outer glow ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.06, 0.11, 32]} />
            <meshBasicMaterial
              color="#ff9f40"
              transparent
              opacity={0.35}
              side={THREE.DoubleSide}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
          {/* Secondary glow */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.1, 0.16, 32]} />
            <meshBasicMaterial
              color="#ffcc80"
              transparent
              opacity={0.15}
              side={THREE.DoubleSide}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
          {/* Inner bright dot */}
          <mesh ref={(el) => { if (el) hotspotRefs.current[i] = el; }}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshBasicMaterial color="#ffd699" transparent opacity={1} />
          </mesh>
        </group>
      ))}

      {/* Bezier arcs connecting metros */}
      <BezierArcs positions={metroPositions} />

      {/* Orbital rings */}
      <OrbitalRings radius={radius} />
    </group>
  );
}

// Bezier arc connections
function BezierArcs({ positions }: { positions: THREE.Vector3[] }) {
  const connections = useMemo(() => {
    const pairs = [
      [0, 1], [0, 4], [1, 3], [2, 4], [3, 0], [4, 5],
    ];

    return pairs.map(([from, to]) => {
      const start = positions[from];
      const end = positions[to];
      const mid = new THREE.Vector3()
        .addVectors(start, end)
        .multiplyScalar(0.5)
        .normalize()
        .multiplyScalar(3.0);

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const points = curve.getPoints(60);
      return points.map(p => [p.x, p.y, p.z] as [number, number, number]);
    });
  }, [positions]);

  return (
    <>
      {connections.map((points, i) => (
        <Line
          key={i}
          points={points}
          color="#ff9f40"
          lineWidth={1}
          transparent
          opacity={0.25}
        />
      ))}
    </>
  );
}

// Animated orbital rings - cleaner style
function OrbitalRings({ radius }: { radius: number }) {
  const ring1Ref = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const ring3Ref = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ring1Ref.current) ring1Ref.current.rotation.z = t * 0.06;
    if (ring2Ref.current) ring2Ref.current.rotation.x = t * 0.04;
    if (ring3Ref.current) ring3Ref.current.rotation.y = t * 0.03;
  });

  return (
    <>
      {/* Primary ring - most visible */}
      <mesh ref={ring1Ref} rotation={[Math.PI / 2.3, 0.15, 0]}>
        <torusGeometry args={[radius * 1.32, 0.005, 8, 150]} />
        <meshBasicMaterial color="#ffb366" transparent opacity={0.25} />
      </mesh>
      {/* Secondary ring */}
      <mesh ref={ring2Ref} rotation={[Math.PI / 3, Math.PI / 4.5, 0]}>
        <torusGeometry args={[radius * 1.42, 0.004, 8, 150]} />
        <meshBasicMaterial color="#ff9f40" transparent opacity={0.18} />
      </mesh>
      {/* Tertiary ring - subtle */}
      <mesh ref={ring3Ref} rotation={[Math.PI / 4.5, -Math.PI / 5, 0]}>
        <torusGeometry args={[radius * 1.52, 0.003, 8, 150]} />
        <meshBasicMaterial color="#ffd699" transparent opacity={0.12} />
      </mesh>
    </>
  );
}

// Camera controller
function CameraController() {
  const { camera, size } = useThree();

  useEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.aspect = size.width / size.height;
      camera.updateProjectionMatrix();
    }
  }, [camera, size]);

  return null;
}

// Globe scene
function GlobeScene() {
  return (
    <>
      <CameraController />
      <ambientLight intensity={0.3} />
      <ParticleGlobe />
    </>
  );
}

// Mini fan chart SVG with animation
function MiniFanChart() {
  return (
    <svg width="100" height="36" viewBox="0 0 100 36" className="opacity-90">
      <defs>
        <linearGradient id="miniFanGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#ff9f40" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#ff6b35" stopOpacity="0.3" />
        </linearGradient>
      </defs>
      {/* P10-P90 band */}
      <path
        d="M0 30 Q25 28 50 22 T100 14 L100 22 Q75 26 50 28 T0 32 Z"
        fill="url(#miniFanGrad)"
      />
      {/* P25-P75 band */}
      <path
        d="M0 28 Q25 26 50 20 T100 16 L100 20 Q75 24 50 26 T0 30 Z"
        fill="rgba(255,159,64,0.4)"
      />
      {/* Median line */}
      <path
        d="M0 29 Q25 27 50 21 T100 18"
        fill="none"
        stroke="#ff9f40"
        strokeWidth="2.5"
        strokeLinecap="round"
        className="animate-fan-draw"
      />
    </svg>
  );
}

// Enhanced Glass Card with Reflect.app style
function GlassCard({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (cardRef.current) {
      gsap.fromTo(
        cardRef.current,
        { opacity: 0, y: 40, scale: 0.92 },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 1,
          delay: 1.8 + delay,
          ease: "power3.out",
        }
      );
    }
  }, [delay]);

  return (
    <div
      ref={cardRef}
      className={`
        relative overflow-hidden rounded-2xl
        bg-gradient-to-br from-white/[0.08] to-white/[0.02]
        backdrop-blur-xl
        border border-white/[0.1]
        shadow-[0_8px_32px_rgba(0,0,0,0.4)]
        p-5
        ${className}
      `}
      style={{ opacity: 0 }}
    >
      {/* Subtle inner glow */}
      <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-transparent pointer-events-none" />
      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}

export function HeroSection() {
  const containerRef = useRef<HTMLElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

      gsap.set(
        [badgeRef.current, labelRef.current, headlineRef.current, subtitleRef.current, ctaRef.current, statsRef.current],
        { opacity: 0, y: 60 }
      );

      tl.to(badgeRef.current, { opacity: 1, y: 0, duration: 0.8, delay: 0.3 })
        .to(headlineRef.current, { opacity: 1, y: 0, duration: 1.2 }, "-=0.5")
        .to(subtitleRef.current, { opacity: 1, y: 0, duration: 1 }, "-=0.8")
        .to(ctaRef.current, { opacity: 1, y: 0, duration: 0.9 }, "-=0.6")
        .to(statsRef.current, { opacity: 1, y: 0, duration: 0.8 }, "-=0.4");
    }, containerRef);

    return () => ctx.revert();
  }, [mounted]);

  return (
    <>
      {/* Cosmic background effects - client-side only to avoid hydration mismatch */}
      {mounted && (
        <>
          {/* Deep cosmic background - Reflect style */}
          <div className="fixed inset-0 -z-30 bg-[#030306]" />

          {/* Primary glow orb - warm amber behind globe */}
          <div
            className="fixed -z-20 pointer-events-none glow-orb"
            style={{
              right: "5%",
              top: "10%",
              width: "700px",
              height: "700px",
              background: `
                radial-gradient(circle at center,
                  rgba(255,159,64,0.25) 0%,
                  rgba(255,120,50,0.15) 30%,
                  rgba(255,80,30,0.08) 50%,
                  transparent 70%
                )
              `,
            }}
          />

          {/* Secondary glow - purple accent */}
          <div
            className="fixed -z-20 pointer-events-none"
            style={{
              left: "-10%",
              bottom: "10%",
              width: "500px",
              height: "500px",
              background: "radial-gradient(circle at center, rgba(139,92,246,0.1) 0%, transparent 60%)",
              filter: "blur(40px)",
            }}
          />

          {/* Top ambient glow */}
          <div
            className="fixed -z-20 pointer-events-none"
            style={{
              left: "30%",
              top: "-20%",
              width: "600px",
              height: "400px",
              background: "radial-gradient(ellipse at center, rgba(255,159,64,0.06) 0%, transparent 60%)",
            }}
          />

          {/* Light beam effect - Reflect style */}
          <div
            className="fixed -z-15 pointer-events-none hidden lg:block"
            style={{
              right: "35%",
              top: "0",
              width: "2px",
              height: "100vh",
              background: "linear-gradient(to bottom, transparent, rgba(255,200,150,0.15) 30%, rgba(255,159,64,0.1) 60%, transparent 90%)",
              filter: "blur(2px)",
            }}
          />
        </>
      )}

      {/* Grain overlay */}
      <div className="grain-overlay" />

      <section
        ref={containerRef}
        className="relative min-h-screen w-full flex flex-col overflow-hidden"
      >
        {/* Header - Stripe Sessions style */}
        <header className="relative z-20 flex items-center justify-between px-8 md:px-12 lg:px-16 py-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-400 via-orange-500 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <span className="text-white font-bold text-sm">L</span>
            </div>
            <span className="text-white font-display font-bold text-xl tracking-tight">
              Lasso
            </span>
          </div>

          {/* Pill-shaped nav - Reflect style */}
          <nav className="hidden md:flex items-center">
            <div className="flex items-center gap-1 px-2 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] backdrop-blur-sm">
              {["Platform", "Research", "Enterprise"].map((item) => (
                <a
                  key={item}
                  href="#"
                  className="px-4 py-2 text-white/50 hover:text-white text-sm font-medium transition-all duration-300 rounded-full hover:bg-white/[0.05]"
                >
                  {item}
                </a>
              ))}
            </div>
          </nav>

          <div className="flex items-center gap-4">
            <a href="#" className="hidden md:block text-white/50 hover:text-white text-sm font-medium transition-colors">
              Login
            </a>
            <Link
              href="/simulate"
              className="px-5 py-2.5 bg-white text-black text-sm font-semibold rounded-full hover:bg-white/90 transition-all duration-300 hover:scale-[1.02]"
            >
              Request Demo
            </Link>
          </div>
        </header>

        {/* Main content */}
        <main className="relative z-10 flex-1 flex items-center px-8 md:px-12 lg:px-16 xl:px-24 pb-24">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-[1600px] mx-auto">
            {/* Left column - Text */}
            <div className="flex flex-col justify-center lg:pr-8">
              {/* Pill Badge - Reflect style */}
              <div ref={badgeRef} className="mb-8">
                <span className="pill-badge pill-badge-accent">
                  <svg className="w-4 h-4 sparkle-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0L14.59 8.41L23 11L14.59 13.59L12 22L9.41 13.59L1 11L9.41 8.41L12 0Z" />
                  </svg>
                  <span>For Loan Officers & Banks</span>
                </span>
              </div>

              {/* Headline - Tempo + Reflect style */}
              <h1
                ref={headlineRef}
                className="font-display text-[clamp(3rem,8vw,5.5rem)] font-bold leading-[0.92] tracking-[-0.04em] mb-8"
              >
                <span className="block text-gradient-accent drop-shadow-[0_0_60px_rgba(255,159,64,0.25)]">
                  Underwrite gig
                </span>
                <span className="block text-gradient-accent drop-shadow-[0_0_60px_rgba(255,159,64,0.25)]">
                  workers with
                </span>
                <span className="block text-white mt-2 text-[0.85em]">confidence.</span>
              </h1>

              {/* Subtitle - improved hierarchy */}
              <div ref={subtitleRef} className="max-w-lg mb-12">
                <p className="text-xl text-white/40 leading-relaxed font-light">
                  Monte Carlo simulations reveal what FICO misses—
                  <span className="text-white/70 font-normal">
                    giving your team the tools to assess gig economy applicants with precision.
                  </span>
                </p>
              </div>

              {/* CTA Group - Enhanced */}
              <div ref={ctaRef} className="flex flex-wrap items-center gap-4">
                <Link href="/simulate" className="btn-primary group">
                  <span>Try Risk Console</span>
                  <svg
                    className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2.5}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Link>
                <button className="btn-secondary group">
                  <span>Schedule Demo</span>
                  <svg className="w-4 h-4 opacity-50 group-hover:opacity-100 transition-opacity" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                    <line x1="16" y1="2" x2="16" y2="6" />
                    <line x1="8" y1="2" x2="8" y2="6" />
                    <line x1="3" y1="10" x2="21" y2="10" />
                  </svg>
                </button>
                <span className="text-xs text-white/20 font-medium tracking-wide ml-2">
                  HooHacks 2026
                </span>
              </div>
            </div>

            {/* Right column - Globe + Cards */}
            <div className="relative h-[500px] lg:h-[650px]">
              {/* Globe */}
              {mounted && (
                <div className="absolute inset-0">
                  <Canvas
                    camera={{ position: [0, 0, 6], fov: 50 }}
                    gl={{ antialias: true, alpha: true }}
                    dpr={[1, 2]}
                  >
                    <Suspense fallback={null}>
                      <GlobeScene />
                    </Suspense>
                  </Canvas>
                </div>
              )}

              {/* Floating Glass Cards - positioned outside globe area */}
              <div className="absolute inset-0 pointer-events-none z-10">
                {/* Card 1: Applicant Assessment - positioned to left of globe */}
                <GlassCard
                  className="absolute top-[8%] -left-[15%] w-[220px] animate-float pointer-events-auto"
                  delay={0}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[10px] font-medium uppercase tracking-widest text-white/40">
                      Applicant #4821
                    </span>
                    <span className="px-2 py-0.5 text-[8px] font-bold uppercase tracking-wide bg-emerald-400/15 text-emerald-400 rounded-md border border-emerald-400/20">
                      Approve
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-display font-bold text-white tracking-tight">
                      3.2%
                    </span>
                    <span className="text-xs text-white/40">P(Default)</span>
                  </div>
                </GlassCard>

                {/* Card 2: Risk Assessment - positioned to right of globe */}
                <GlassCard
                  className="absolute top-[35%] -right-[10%] w-[200px] animate-float-delayed pointer-events-auto"
                  delay={0.15}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] font-medium uppercase tracking-widest text-white/40">
                      Risk Analysis
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="relative w-2 h-2">
                      <div className="absolute inset-0 rounded-full bg-amber-400 animate-ping opacity-50" />
                      <div className="absolute inset-0 rounded-full bg-amber-400" />
                    </div>
                    <span className="text-[11px] text-white/60 font-medium">
                      Running 5,000 paths
                    </span>
                  </div>
                  <MiniFanChart />
                </GlassCard>

                {/* Card 3: Stress Test Results - positioned below left */}
                <GlassCard
                  className="absolute bottom-[12%] -left-[10%] w-[220px] animate-float-delayed-2 pointer-events-auto"
                  delay={0.3}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[10px] font-medium uppercase tracking-widest text-white/40">
                      Gas Spike Stress
                    </span>
                    <span className="px-2 py-0.5 text-[8px] font-bold uppercase tracking-wide bg-red-400/15 text-red-400 rounded-md border border-red-400/20">
                      Decline
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-display font-bold text-white tracking-tight">
                      18.1%
                    </span>
                    <span className="text-xs text-white/40">P(Default)</span>
                  </div>
                </GlassCard>
              </div>
            </div>
          </div>
        </main>

        {/* Bottom stats bar - Tempo numbered style */}
        <div
          ref={statsRef}
          className="relative z-20 border-t border-white/[0.04] bg-gradient-to-r from-black/40 via-black/20 to-black/40 backdrop-blur-xl"
        >
          <div className="flex flex-col lg:flex-row items-center justify-between px-8 md:px-12 lg:px-16 py-6 gap-6">
            {/* Stats - Tempo numbered style */}
            <div className="flex items-center gap-6 md:gap-10">
              {[
                { num: "01", value: "5,000", label: "Simulations" },
                { num: "02", value: "12", label: "Data Points" },
                { num: "03", value: "6", label: "Metro Models" },
                { num: "04", value: "<2s", label: "Assessment" },
              ].map((stat, i) => (
                <div key={i} className="flex items-center gap-6 md:gap-10">
                  {i > 0 && <div className="w-px h-10 bg-gradient-to-b from-transparent via-white/10 to-transparent" />}
                  <div className="flex items-baseline gap-3">
                    <span className="text-[10px] font-mono text-white/20">{stat.num}</span>
                    <div>
                      <div className="text-2xl md:text-3xl font-display font-bold text-white tracking-tight">
                        {stat.value}
                      </div>
                      <div className="text-[10px] text-white/30 uppercase tracking-[0.15em] font-medium">
                        {stat.label}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Sources - more subtle */}
            <div className="flex items-center gap-4">
              <span className="text-[10px] text-white/20 uppercase tracking-wider font-medium">
                Data from
              </span>
              <div className="flex items-center gap-3">
                {["JPMorgan Chase", "Federal Reserve", "Gridwise"].map((source) => (
                  <span
                    key={source}
                    className="text-[10px] text-white/30 hover:text-white/50 transition-colors cursor-default whitespace-nowrap font-medium"
                  >
                    {source}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-30">
          <span className="text-[9px] text-white/50 uppercase tracking-[0.2em] font-medium">
            Scroll
          </span>
          <div className="w-px h-10 bg-gradient-to-b from-white/30 to-transparent" />
        </div>
      </section>
    </>
  );
}

export default HeroSection;
