import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge class names with Tailwind CSS conflict resolution
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number as currency (USD)
 */
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Format a number as a percentage
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a large number with abbreviations (K, M)
 */
export function formatCompact(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    compactDisplay: "short",
  }).format(value);
}

/**
 * Calculate monthly loan payment using standard amortization formula
 */
export function calculateMonthlyPayment(
  principal: number,
  annualRate: number,
  termMonths: number
): number {
  if (annualRate === 0) return principal / termMonths;

  const monthlyRate = annualRate / 12;
  const payment =
    (principal * (monthlyRate * Math.pow(1 + monthlyRate, termMonths))) /
    (Math.pow(1 + monthlyRate, termMonths) - 1);

  return Math.round(payment * 100) / 100;
}

/**
 * Get risk level based on default probability
 */
export function getRiskLevel(
  defaultProbability: number
): "low" | "medium" | "high" {
  if (defaultProbability < 0.05) return "low";
  if (defaultProbability < 0.15) return "medium";
  return "high";
}

/**
 * Get color class based on risk level
 */
export function getRiskColorClass(riskLevel: "low" | "medium" | "high"): string {
  switch (riskLevel) {
    case "low":
      return "text-secondary";
    case "medium":
      return "text-warning";
    case "high":
      return "text-danger";
  }
}
