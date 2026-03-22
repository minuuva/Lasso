"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useSimulation } from "@/contexts/SimulationContext";
import {
  PRESET_SCENARIOS,
  type PresetScenarioId,
} from "@/types";
import { formatPercent } from "@/lib/utils";
import {
  AlertTriangle,
  TrendingDown,
  Fuel,
  DollarSign,
  Heart,
  X,
  Zap,
} from "lucide-react";

// Icon mapping for preset scenarios
const scenarioIcons: Record<PresetScenarioId, React.ReactNode> = {
  recession_2008: <TrendingDown className="h-4 w-4" />,
  recession_2020: <AlertTriangle className="h-4 w-4" />,
  gas_spike: <Fuel className="h-4 w-4" />,
  platform_pay_cut: <DollarSign className="h-4 w-4" />,
  injury: <Heart className="h-4 w-4" />,
};

// Color mapping for preset scenarios
const scenarioColors: Record<PresetScenarioId, string> = {
  recession_2008: "bg-danger/10 text-danger hover:bg-danger/20",
  recession_2020: "bg-warning/10 text-warning hover:bg-warning/20",
  gas_spike: "bg-warning/10 text-warning hover:bg-warning/20",
  platform_pay_cut: "bg-danger/10 text-danger hover:bg-danger/20",
  injury: "bg-danger/10 text-danger hover:bg-danger/20",
};

export function ScenarioPanel() {
  const {
    state,
    setScenario,
    runComparison,
    hasResults,
    hasComparison,
  } = useSimulation();
  const { scenario, isLoading, compareResult } = state;

  const handlePresetClick = async (id: PresetScenarioId) => {
    const presetScenario = PRESET_SCENARIOS[id];
    setScenario(presetScenario);

    // Auto-run comparison if we have baseline results
    if (hasResults) {
      // Small delay to ensure state is updated
      setTimeout(() => {
        runComparison();
      }, 100);
    }
  };

  const handleClearScenario = () => {
    setScenario(null);
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Zap className="h-5 w-5 text-warning" />
          Stress Testing
        </CardTitle>
        <CardDescription>
          Test how economic scenarios affect outcomes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Active Scenario Badge */}
        {scenario && (
          <div className="flex items-center justify-between rounded-lg bg-warning/10 p-3">
            <div>
              <p className="text-sm font-medium text-warning">
                {scenario.scenarioName}
              </p>
              <p className="text-xs text-muted-foreground">
                {scenario.description}
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearScenario}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Preset Scenario Buttons */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground uppercase">
            Preset Scenarios
          </p>
          <div className="grid grid-cols-1 gap-2">
            {(Object.keys(PRESET_SCENARIOS) as PresetScenarioId[]).map(
              (id) => {
                const preset = PRESET_SCENARIOS[id];
                const isActive = scenario?.scenarioName === preset.scenarioName;

                return (
                  <Button
                    key={id}
                    variant="ghost"
                    className={`justify-start gap-2 h-auto py-2 px-3 ${
                      isActive
                        ? "bg-warning/20 text-warning"
                        : scenarioColors[id]
                    }`}
                    onClick={() => handlePresetClick(id)}
                    disabled={isLoading || isActive}
                  >
                    {scenarioIcons[id]}
                    <div className="text-left">
                      <p className="text-sm font-medium">
                        {preset.scenarioName}
                      </p>
                      <p className="text-xs opacity-80">
                        {preset.durationMonths} months
                      </p>
                    </div>
                  </Button>
                );
              }
            )}
          </div>
        </div>

        {/* Delta Summary */}
        {hasComparison && compareResult && (
          <div className="space-y-3 pt-4 border-t border-border">
            <p className="text-xs font-medium text-muted-foreground uppercase">
              Impact Summary
            </p>

            {/* Default Probability Change */}
            <div className="flex items-center justify-between rounded-lg bg-muted p-3">
              <span className="text-sm">Default Probability Change</span>
              <Badge
                variant={
                  compareResult.delta.defaultProbChange > 0.05
                    ? "destructive"
                    : compareResult.delta.defaultProbChange > 0.02
                    ? "warning"
                    : "muted"
                }
              >
                {compareResult.delta.defaultProbChange > 0 ? "+" : ""}
                {formatPercent(compareResult.delta.defaultProbChange)}
              </Badge>
            </div>

            {/* Income Change */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Median Income Change
              </span>
              <span
                className={`text-sm font-medium ${
                  compareResult.delta.medianIncomeChange < 0
                    ? "text-danger"
                    : "text-secondary"
                }`}
              >
                {compareResult.delta.medianIncomeChange > 0 ? "+" : ""}$
                {compareResult.delta.medianIncomeChange.toLocaleString()}/mo
              </span>
            </div>

            {/* Savings Depletion Change */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Savings Depletion Risk
              </span>
              <span
                className={`text-sm font-medium ${
                  compareResult.delta.savingsDepletedChange > 0.05
                    ? "text-danger"
                    : "text-muted-foreground"
                }`}
              >
                {compareResult.delta.savingsDepletedChange > 0 ? "+" : ""}
                {formatPercent(compareResult.delta.savingsDepletedChange)}
              </span>
            </div>
          </div>
        )}

        {/* Help text */}
        {!hasResults && (
          <p className="text-xs text-muted-foreground text-center pt-4">
            Run a baseline simulation first to enable stress testing
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export default ScenarioPanel;
