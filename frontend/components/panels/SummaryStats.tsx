"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { useSimulation } from "@/contexts/SimulationContext";
import { formatCurrency, formatPercent } from "@/lib/utils";
import {
  DollarSign,
  TrendingDown,
  PiggyBank,
  AlertTriangle,
} from "lucide-react";

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtext?: string;
  variant?: "default" | "warning" | "danger" | "success";
}

function StatCard({
  icon,
  label,
  value,
  subtext,
  variant = "default",
}: StatCardProps) {
  const colorClasses = {
    default: "text-primary",
    success: "text-secondary",
    warning: "text-warning",
    danger: "text-danger",
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {label}
            </p>
            <p className={`mt-1 text-2xl font-bold ${colorClasses[variant]}`}>
              {value}
            </p>
            {subtext && (
              <p className="mt-1 text-xs text-muted-foreground">{subtext}</p>
            )}
          </div>
          <div className={`p-2 rounded-lg bg-muted ${colorClasses[variant]}`}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function SummaryStats() {
  const { state, hasResults, hasComparison } = useSimulation();

  const summary = hasComparison
    ? state.compareResult?.baseline.summary
    : state.baselineResult?.summary;

  if (!hasResults && !hasComparison) {
    return (
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="h-16 flex items-center justify-center text-muted-foreground text-sm">
                Run simulation to see stats
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!summary) return null;

  // Calculate risk variant for negative cash flow
  const negCashFlowVariant =
    summary.probNegativeCashFlowAnyMonth > 0.3
      ? "danger"
      : summary.probNegativeCashFlowAnyMonth > 0.15
      ? "warning"
      : "success";

  // Calculate runway in months
  const runway =
    summary.medianMonthlyIncome > 0
      ? Math.round(
          state.profile.currentSavings /
            (state.profile.monthlyRent +
              state.profile.monthlyFixedExpenses -
              summary.medianMonthlyIncome * 0.2)
        )
      : 0;

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      <StatCard
        icon={<DollarSign className="h-5 w-5" />}
        label="Median Monthly Income"
        value={formatCurrency(summary.medianMonthlyIncome)}
        subtext={`Mean: ${formatCurrency(summary.meanMonthlyIncome)}`}
        variant="default"
      />

      <StatCard
        icon={<TrendingDown className="h-5 w-5" />}
        label="Income Volatility"
        value={formatPercent(summary.incomeVolatilityCv)}
        subtext="Coefficient of variation"
        variant={
          summary.incomeVolatilityCv > 0.4
            ? "warning"
            : summary.incomeVolatilityCv > 0.3
            ? "default"
            : "success"
        }
      />

      <StatCard
        icon={<AlertTriangle className="h-5 w-5" />}
        label="Negative Cash Flow Risk"
        value={formatPercent(summary.probNegativeCashFlowAnyMonth)}
        subtext="Chance of any negative month"
        variant={negCashFlowVariant}
      />

      <StatCard
        icon={<PiggyBank className="h-5 w-5" />}
        label="Savings at End"
        value={formatCurrency(summary.medianSavingsAtEnd)}
        subtext={`Worst case (P10): ${formatCurrency(summary.worstMonthP10)}`}
        variant={
          summary.probSavingsDepleted > 0.2
            ? "danger"
            : summary.probSavingsDepleted > 0.1
            ? "warning"
            : "success"
        }
      />
    </div>
  );
}

export default SummaryStats;
