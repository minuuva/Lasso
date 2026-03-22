"use client";

import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useSimulation } from "@/contexts/SimulationContext";
import { percentilesToChartData } from "@/types";
import { formatCurrency } from "@/lib/utils";
import { CHART_COLORS } from "@/lib/constants";
import { TrendingUp } from "lucide-react";

interface FanChartProps {
  title?: string;
  dataType?: "income" | "cashFlow" | "savings";
}

export function FanChart({
  title = "Monthly Net Income Projection",
  dataType = "income",
}: FanChartProps) {
  const { state, hasResults, hasComparison } = useSimulation();
  const { baselineResult, compareResult, isLoading } = state;

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!hasResults && !hasComparison) return [];

    const baseline = hasComparison
      ? compareResult?.baseline
      : baselineResult;
    const stressed = hasComparison ? compareResult?.stressed : null;

    if (!baseline) return [];

    // Select the right percentiles based on data type
    const getPercentiles = (result: typeof baseline) => {
      switch (dataType) {
        case "income":
          return result.incomePercentiles;
        case "cashFlow":
          return result.cashFlowPercentiles;
        case "savings":
          return result.savingsPercentiles;
      }
    };

    const baselinePercentiles = getPercentiles(baseline);
    const stressedPercentiles = stressed
      ? getPercentiles(stressed)
      : undefined;

    return percentilesToChartData(baselinePercentiles, stressedPercentiles);
  }, [hasResults, hasComparison, baselineResult, compareResult, dataType]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; name: string; color: string }>;
    label?: string;
  }) => {
    if (!active || !payload || !payload.length) return null;

    // Find relevant values
    const p50 = payload.find((p) => p.name === "Median")?.value ?? 0;
    const p10 = payload.find((p) => p.name === "p10-p25")?.value ?? 0;
    const p90 = payload.find((p) => p.name === "p75-p90")?.value ?? 0;

    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
        <p className="font-medium text-card-foreground">{label}</p>
        <div className="mt-2 space-y-1 text-sm">
          <p className="text-primary">
            Median: {formatCurrency(p50)}
          </p>
          <p className="text-muted-foreground">
            Range: {formatCurrency(p10)} - {formatCurrency(p90)}
          </p>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-80 items-center justify-center">
            <div className="text-center">
              <div className="mb-4 animate-pulse text-lg text-muted-foreground">
                Running {state.config.numPaths.toLocaleString()} simulations...
              </div>
              <Skeleton className="mx-auto h-4 w-48" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!hasResults && !hasComparison) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-80 items-center justify-center">
            <div className="text-center text-muted-foreground">
              <TrendingUp className="mx-auto mb-4 h-12 w-12 opacity-30" />
              <p>Enter a profile and run a simulation to see projections</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-80 w-full" style={{ minHeight: 320 }}>
          <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={200}>
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                {/* Baseline gradients */}
                <linearGradient id="colorP90" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0066CC" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#0066CC" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="colorP75" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0066CC" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#0066CC" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0066CC" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#0066CC" stopOpacity={0.2} />
                </linearGradient>
                {/* Stressed gradients */}
                <linearGradient id="colorStressP50" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#D32F2F" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#D32F2F" stopOpacity={0.1} />
                </linearGradient>
              </defs>

              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />

              <XAxis
                dataKey="monthLabel"
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                interval="preserveStartEnd"
                tickFormatter={(value, index) => {
                  // Show every 6th month label
                  if (index % 6 === 0) return value;
                  return "";
                }}
              />

              <YAxis
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                tickFormatter={(value) =>
                  `$${(value / 1000).toFixed(0)}k`
                }
                width={60}
              />

              <Tooltip content={<CustomTooltip />} />

              {/* Baseline bands - stacked from bottom to top */}
              {/* p10-p25 band */}
              <Area
                type="monotone"
                dataKey="p10"
                stackId="baseline"
                stroke="none"
                fill="url(#colorP90)"
                name="p10-p25"
                animationDuration={1000}
              />
              {/* p25-p50 band */}
              <Area
                type="monotone"
                dataKey="p25"
                stackId="baseline2"
                stroke="none"
                fill="url(#colorP75)"
                name="p25-p50"
                animationDuration={1000}
              />
              {/* Median line */}
              <Area
                type="monotone"
                dataKey="p50"
                stackId="baseline3"
                stroke={CHART_COLORS.fan.median}
                strokeWidth={2}
                fill="url(#colorP50)"
                name="Median"
                animationDuration={1000}
              />
              {/* p50-p75 band */}
              <Area
                type="monotone"
                dataKey="p75"
                stackId="baseline4"
                stroke="none"
                fill="url(#colorP75)"
                name="p50-p75"
                animationDuration={1000}
              />
              {/* p75-p90 band */}
              <Area
                type="monotone"
                dataKey="p90"
                stackId="baseline5"
                stroke="none"
                fill="url(#colorP90)"
                name="p75-p90"
                animationDuration={1000}
              />

              {/* Stressed scenario overlay (if comparing) */}
              {hasComparison && (
                <Area
                  type="monotone"
                  dataKey="stressedP50"
                  stroke={CHART_COLORS.stress.median}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  fill="url(#colorStressP50)"
                  name="Stressed Median"
                  animationDuration={1000}
                />
              )}

              <Legend
                wrapperStyle={{ paddingTop: 10 }}
                formatter={(value) => (
                  <span className="text-sm text-muted-foreground">{value}</span>
                )}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Legend explanation */}
        <div className="mt-4 flex items-center justify-center gap-6 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: CHART_COLORS.fan.median }}
            />
            <span>Median (50th percentile)</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="h-3 w-6 rounded opacity-30"
              style={{ backgroundColor: "#0066CC" }}
            />
            <span>10th-90th percentile range</span>
          </div>
          {hasComparison && (
            <div className="flex items-center gap-2">
              <div
                className="h-0.5 w-6 border-dashed border-b-2"
                style={{ borderColor: CHART_COLORS.stress.median }}
              />
              <span>Stressed scenario</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default FanChart;
