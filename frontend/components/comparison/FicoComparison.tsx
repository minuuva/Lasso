"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import type { LoanEvaluation } from "@/types";
import {
  getRiskAssessmentColor,
  getRiskAssessmentLabel,
  getFicoRatingLabel,
} from "@/types";
import { formatPercent } from "@/lib/utils";
import { ArrowDown, ArrowUp, Minus, Shield, AlertTriangle } from "lucide-react";

interface FicoComparisonProps {
  evaluation: LoanEvaluation;
}

export function FicoComparison({ evaluation }: FicoComparisonProps) {
  const {
    ficoScore,
    ficoEstimatedDefaultRate,
    probDefault,
    riskDelta,
    riskAssessment,
  } = evaluation;

  // Calculate delta percentage
  const deltaPercent = Math.abs(riskDelta * 100);
  const isSafer = riskDelta < 0;
  const isAligned = riskAssessment === "ALIGNED";

  // Get badge variant
  const badgeVariant =
    riskAssessment === "FICO_OVERESTIMATES_RISK"
      ? "success"
      : riskAssessment === "ALIGNED"
      ? "warning"
      : "destructive";

  return (
    <div className="space-y-3">
      {/* Comparison Header */}
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-sm">FICO Comparison</h4>
        <Badge variant={badgeVariant} className="text-xs">
          {getRiskAssessmentLabel(riskAssessment)}
        </Badge>
      </div>

      {/* Side by Side Cards */}
      <div className="grid grid-cols-2 gap-3">
        {/* FICO Model */}
        <div className="rounded-lg border border-border p-3">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs font-medium text-muted-foreground">
              FICO Model
            </span>
          </div>
          <p className="text-lg font-bold">
            {formatPercent(ficoEstimatedDefaultRate)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Score: {ficoScore} ({getFicoRatingLabel(ficoScore)})
          </p>
        </div>

        {/* Our Model */}
        <div className="rounded-lg border border-primary/50 bg-primary/5 p-3">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4 text-primary" />
            <span className="text-xs font-medium text-primary">Our Model</span>
          </div>
          <p className="text-lg font-bold text-primary">
            {formatPercent(probDefault)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Monte Carlo simulation
          </p>
        </div>
      </div>

      {/* Delta Indicator */}
      <div
        className={`flex items-center justify-center gap-2 rounded-lg p-3 ${
          isAligned
            ? "bg-warning/10 text-warning"
            : isSafer
            ? "bg-secondary/10 text-secondary"
            : "bg-danger/10 text-danger"
        }`}
      >
        {isAligned ? (
          <Minus className="h-5 w-5" />
        ) : isSafer ? (
          <ArrowDown className="h-5 w-5" />
        ) : (
          <ArrowUp className="h-5 w-5" />
        )}
        <span className="font-medium">
          {isAligned
            ? "Risk assessment aligned with FICO"
            : isSafer
            ? `${deltaPercent.toFixed(1)}% safer than FICO predicts`
            : `${deltaPercent.toFixed(1)}% riskier than FICO predicts`}
        </span>
      </div>

      {/* Explanation */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        {riskAssessment === "FICO_OVERESTIMATES_RISK"
          ? "Our model suggests this borrower has lower default risk than traditional FICO scoring indicates. Factors like stable gig income patterns, adequate savings, and experience contribute to this assessment."
          : riskAssessment === "ALIGNED"
          ? "Our Monte Carlo simulation aligns with traditional FICO risk assessment for this borrower profile."
          : "Our model identifies higher risk than FICO suggests. Income volatility, limited savings buffer, or other factors may increase default probability beyond what credit score alone indicates."}
      </p>
    </div>
  );
}

export default FicoComparison;
