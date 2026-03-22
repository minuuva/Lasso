/**
 * Loan parameters - matches backend LoanParams
 */
export interface LoanParams {
  amount: number; // e.g., 15000
  annualRate: number; // e.g., 0.072 for 7.2%
  termMonths: number; // 24, 36, 48, or 60
}

/**
 * Loan evaluation result - matches backend LoanEvaluation
 */
export interface LoanEvaluation {
  loanAmount: number;
  annualRate: number;
  termMonths: number;
  monthlyPayment: number;

  // Our model's assessment (Monte Carlo / cash-flow based — no credit score)
  probMissOnePayment: number;
  probMissThreeConsecutive: number;
  probDefault: number;
  monthsToFirstMissP50: number;
}

/**
 * Default loan parameters
 */
export const DEFAULT_LOAN_PARAMS: LoanParams = {
  amount: 15000,
  annualRate: 0.072,
  termMonths: 60,
};

/**
 * Available loan term options (in months)
 */
export const LOAN_TERM_OPTIONS = [24, 36, 48, 60] as const;
