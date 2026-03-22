"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useSimulation } from "@/contexts/SimulationContext";
import { LOAN_TERM_OPTIONS } from "@/types";
import { formatCurrency, formatPercent, calculateMonthlyPayment } from "@/lib/utils";
import { CreditCard, DollarSign, Percent, Clock } from "lucide-react";

export function LoanEvalPanel() {
  const { state, updateLoanParams, hasResults, hasComparison } = useSimulation();
  const { loanParams } = state;

  // Get loan evaluation from results
  const loanEval = hasComparison
    ? state.compareResult?.baseline.loanEvaluation
    : state.baselineResult?.loanEvaluation;

  // Calculate monthly payment
  const monthlyPayment = calculateMonthlyPayment(
    loanParams.amount,
    loanParams.annualRate,
    loanParams.termMonths
  );

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <CreditCard className="h-5 w-5 text-primary" />
          Loan Evaluation
        </CardTitle>
        <CardDescription>
          Evaluate loan affordability with Monte Carlo analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Loan Inputs */}
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="amount" className="flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              Loan Amount
            </Label>
            <Input
              id="amount"
              type="number"
              value={loanParams.amount}
              onChange={(e) =>
                updateLoanParams({
                  amount: Math.max(0, parseInt(e.target.value) || 0),
                })
              }
              min={0}
              max={100000}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="rate" className="flex items-center gap-2">
              <Percent className="h-4 w-4" />
              Annual Interest Rate (%)
            </Label>
            <Input
              id="rate"
              type="number"
              step="0.1"
              value={(loanParams.annualRate * 100).toFixed(1)}
              onChange={(e) =>
                updateLoanParams({
                  annualRate: Math.max(0, parseFloat(e.target.value) || 0) / 100,
                })
              }
              min={0}
              max={30}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="term" className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Loan Term
            </Label>
            <Select
              value={loanParams.termMonths.toString()}
              onValueChange={(value) =>
                updateLoanParams({ termMonths: parseInt(value) })
              }
            >
              <SelectTrigger id="term">
                <SelectValue placeholder="Select term" />
              </SelectTrigger>
              <SelectContent>
                {LOAN_TERM_OPTIONS.map((months) => (
                  <SelectItem key={months} value={months.toString()}>
                    {months} months ({Math.round(months / 12)} years)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Monthly Payment Display */}
        <div className="rounded-lg bg-muted p-4">
          <p className="text-xs font-medium text-muted-foreground uppercase">
            Estimated Monthly Payment
          </p>
          <p className="mt-1 text-3xl font-bold text-primary">
            {formatCurrency(monthlyPayment)}
          </p>
        </div>

        {/* Evaluation Results */}
        {(hasResults || hasComparison) && loanEval && (
          <div className="space-y-4 pt-4 border-t border-border">
            <h4 className="font-medium text-sm">Risk Assessment</h4>

            {/* Default Probability */}
            <div className="rounded-lg bg-muted p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Our Model: Default Probability
                </span>
                <Badge
                  variant={
                    loanEval.probDefault < 0.05
                      ? "success"
                      : loanEval.probDefault < 0.15
                      ? "warning"
                      : "destructive"
                  }
                >
                  {formatPercent(loanEval.probDefault)}
                </Badge>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Miss 1+ Payment
                </span>
                <span className="text-sm font-medium">
                  {formatPercent(loanEval.probMissOnePayment)}
                </span>
              </div>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  90+ Day Delinquency
                </span>
                <span className="text-sm font-medium">
                  {formatPercent(loanEval.probMissThreeConsecutive)}
                </span>
              </div>
            </div>
          </div>
        )}

        {!hasResults && !hasComparison && (
          <div className="pt-4 text-center text-sm text-muted-foreground">
            Run a simulation to see loan evaluation
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default LoanEvalPanel;
