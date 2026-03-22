"use client";

import React, { useRef, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// Simplex noise shader for organic, flowing gradients
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  uniform float uTime;
  uniform vec2 uMouse;
  uniform vec2 uResolution;
  varying vec2 vUv;

  // Simplex noise functions
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                           + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m*m;
    m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  // Fractal Brownian Motion
  float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 6; i++) {
      value += amplitude * snoise(p * frequency);
      amplitude *= 0.5;
      frequency *= 2.0;
    }
    return value;
  }

  void main() {
    vec2 uv = vUv;
    vec2 aspect = vec2(uResolution.x / uResolution.y, 1.0);

    // Slow time for smooth movement
    float time = uTime * 0.15;

    // Mouse influence - subtle warp
    vec2 mouseInfluence = (uMouse - 0.5) * 0.1;

    // Create layered noise
    float n1 = fbm(uv * 2.0 + time * 0.3 + mouseInfluence);
    float n2 = fbm(uv * 3.0 - time * 0.2);
    float n3 = fbm(uv * 1.5 + vec2(time * 0.1, -time * 0.15));

    // Combine noise layers
    float noise = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

    // Color palette - deep blacks to warm amber/orange
    vec3 color1 = vec3(0.02, 0.02, 0.04);  // Near black
    vec3 color2 = vec3(0.08, 0.04, 0.12);  // Deep purple-black
    vec3 color3 = vec3(0.4, 0.15, 0.05);   // Deep orange
    vec3 color4 = vec3(0.95, 0.6, 0.2);    // Bright amber

    // Create gradient based on noise and position
    float gradient = uv.y * 0.5 + noise * 0.5;
    gradient = smoothstep(0.0, 1.0, gradient);

    // Mix colors
    vec3 color;
    if (gradient < 0.3) {
      color = mix(color1, color2, gradient / 0.3);
    } else if (gradient < 0.6) {
      color = mix(color2, color3, (gradient - 0.3) / 0.3);
    } else {
      color = mix(color3, color4, (gradient - 0.6) / 0.4);
    }

    // Add subtle grain
    float grain = snoise(uv * 500.0 + time * 10.0) * 0.03;
    color += grain;

    // Vignette
    float vignette = 1.0 - length((uv - 0.5) * 1.2);
    vignette = smoothstep(0.0, 0.7, vignette);
    color *= vignette * 0.8 + 0.2;

    // Add subtle glow spots
    float glow1 = 1.0 - length((uv - vec2(0.7, 0.3)) * vec2(1.5, 2.0));
    glow1 = smoothstep(0.0, 0.5, glow1) * 0.15;
    color += color4 * glow1;

    float glow2 = 1.0 - length((uv - vec2(0.3, 0.7)) * vec2(2.0, 1.5));
    glow2 = smoothstep(0.0, 0.4, glow2) * 0.1;
    color += color3 * glow2;

    gl_FragColor = vec4(color, 1.0);
  }
`;

function ShaderPlane() {
  const meshRef = useRef<THREE.Mesh>(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5 });
  const { viewport, size } = useThree();

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2(0.5, 0.5) },
      uResolution: { value: new THREE.Vector2(size.width, size.height) },
    }),
    [size.width, size.height]
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = {
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      };
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.uTime.value = state.clock.elapsedTime;

      // Smooth mouse lerp
      material.uniforms.uMouse.value.x += (mouseRef.current.x - material.uniforms.uMouse.value.x) * 0.05;
      material.uniforms.uMouse.value.y += (mouseRef.current.y - material.uniforms.uMouse.value.y) * 0.05;
    }
  });

  return (
    <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
      <planeGeometry args={[1, 1, 1, 1]} />
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
      />
    </mesh>
  );
}

export function ShaderBackground({ className = "" }: { className?: string }) {
  return (
    <div className={`fixed inset-0 -z-10 ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 1] }}
        gl={{ antialias: false, alpha: false }}
        dpr={[1, 1.5]}
      >
        <ShaderPlane />
      </Canvas>
    </div>
  );
}

export default ShaderBackground;
