import type { NextConfig } from "next";
import path from "path";
import { loadEnvConfig } from "@next/env";

// Next only loads .env* from this directory by default; repo-root .env.local is ignored otherwise.
const repoRoot = path.join(__dirname, "..");
loadEnvConfig(repoRoot);

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
