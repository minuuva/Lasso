"use client";

import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useMemo,
} from "react";
import type {
  GigWorkerProfile,
  SimulationConfig,
  SimulationResult,
  LoanParams,
  MacroScenario,
  CompareResult,
} from "@/types";
import {
  DEFAULT_PROFILE,
  DEFAULT_SIMULATION_CONFIG,
  DEFAULT_LOAN_PARAMS,
} from "@/types";
import api from "@/lib/api";

// State shape
interface SimulationState {
  profile: GigWorkerProfile;
  config: SimulationConfig;
  loanParams: LoanParams;
  scenario: MacroScenario | null;

  baselineResult: SimulationResult | null;
  compareResult: CompareResult | null;

  isLoading: boolean;
  error: string | null;
}

// Action types
type SimulationAction =
  | { type: "UPDATE_PROFILE"; payload: Partial<GigWorkerProfile> }
  | { type: "UPDATE_CONFIG"; payload: Partial<SimulationConfig> }
  | { type: "UPDATE_LOAN_PARAMS"; payload: Partial<LoanParams> }
  | { type: "SET_SCENARIO"; payload: MacroScenario | null }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_BASELINE_RESULT"; payload: SimulationResult | null }
  | { type: "SET_COMPARE_RESULT"; payload: CompareResult | null }
  | { type: "RESET" };

// Initial state
const initialState: SimulationState = {
  profile: DEFAULT_PROFILE,
  config: DEFAULT_SIMULATION_CONFIG,
  loanParams: DEFAULT_LOAN_PARAMS,
  scenario: null,
  baselineResult: null,
  compareResult: null,
  isLoading: false,
  error: null,
};

// Reducer
function simulationReducer(
  state: SimulationState,
  action: SimulationAction
): SimulationState {
  switch (action.type) {
    case "UPDATE_PROFILE":
      return {
        ...state,
        profile: { ...state.profile, ...action.payload },
        // Clear results when profile changes
        baselineResult: null,
        compareResult: null,
      };
    case "UPDATE_CONFIG":
      return {
        ...state,
        config: { ...state.config, ...action.payload },
        baselineResult: null,
        compareResult: null,
      };
    case "UPDATE_LOAN_PARAMS":
      return {
        ...state,
        loanParams: { ...state.loanParams, ...action.payload },
      };
    case "SET_SCENARIO":
      return {
        ...state,
        scenario: action.payload,
        compareResult: null,
      };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_BASELINE_RESULT":
      return { ...state, baselineResult: action.payload };
    case "SET_COMPARE_RESULT":
      return { ...state, compareResult: action.payload };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

// Context type
interface SimulationContextType {
  state: SimulationState;
  updateProfile: (profile: Partial<GigWorkerProfile>) => void;
  updateConfig: (config: Partial<SimulationConfig>) => void;
  updateLoanParams: (params: Partial<LoanParams>) => void;
  setScenario: (scenario: MacroScenario | null) => void;
  runSimulation: () => Promise<void>;
  runComparison: () => Promise<void>;
  reset: () => void;
  // Computed values
  hasResults: boolean;
  hasComparison: boolean;
  currentResult: SimulationResult | null;
}

// Create context
const SimulationContext = createContext<SimulationContextType | null>(null);

// Provider component
export function SimulationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [state, dispatch] = useReducer(simulationReducer, initialState);

  // Actions
  const updateProfile = useCallback((profile: Partial<GigWorkerProfile>) => {
    dispatch({ type: "UPDATE_PROFILE", payload: profile });
  }, []);

  const updateConfig = useCallback((config: Partial<SimulationConfig>) => {
    dispatch({ type: "UPDATE_CONFIG", payload: config });
  }, []);

  const updateLoanParams = useCallback((params: Partial<LoanParams>) => {
    dispatch({ type: "UPDATE_LOAN_PARAMS", payload: params });
  }, []);

  const setScenario = useCallback((scenario: MacroScenario | null) => {
    dispatch({ type: "SET_SCENARIO", payload: scenario });
  }, []);

  const runSimulation = useCallback(async () => {
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });

    try {
      const result = await api.simulate(
        state.profile,
        state.config,
        state.loanParams
      );
      dispatch({ type: "SET_BASELINE_RESULT", payload: result });
    } catch (error) {
      dispatch({
        type: "SET_ERROR",
        payload:
          error instanceof Error ? error.message : "Simulation failed",
      });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }, [state.profile, state.config, state.loanParams]);

  const runComparison = useCallback(async () => {
    if (!state.scenario) return;

    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });

    try {
      const result = await api.compare(
        state.profile,
        state.config,
        state.loanParams,
        state.scenario
      );
      dispatch({ type: "SET_COMPARE_RESULT", payload: result });
    } catch (error) {
      dispatch({
        type: "SET_ERROR",
        payload:
          error instanceof Error ? error.message : "Comparison failed",
      });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }, [state.profile, state.config, state.loanParams, state.scenario]);

  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  // Computed values
  const hasResults = state.baselineResult !== null;
  const hasComparison = state.compareResult !== null;
  const currentResult = hasComparison
    ? state.compareResult?.baseline ?? null
    : state.baselineResult;

  const value = useMemo(
    () => ({
      state,
      updateProfile,
      updateConfig,
      updateLoanParams,
      setScenario,
      runSimulation,
      runComparison,
      reset,
      hasResults,
      hasComparison,
      currentResult,
    }),
    [
      state,
      updateProfile,
      updateConfig,
      updateLoanParams,
      setScenario,
      runSimulation,
      runComparison,
      reset,
      hasResults,
      hasComparison,
      currentResult,
    ]
  );

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
}

// Hook to use the context
export function useSimulation() {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error(
      "useSimulation must be used within a SimulationProvider"
    );
  }
  return context;
}

/** Same as useSimulation but returns null outside SimulationProvider (e.g. marketing AI chat). */
export function useSimulationOptional(): SimulationContextType | null {
  return useContext(SimulationContext);
}

export default SimulationContext;
