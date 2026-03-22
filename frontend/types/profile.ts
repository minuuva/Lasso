/**
 * Gig platform types - matches backend Platform enum
 */
export enum Platform {
  UBER_RIDESHARE = "uber_rideshare",
  UBER_EATS = "uber_eats",
  DOORDASH = "doordash",
  INSTACART = "instacart",
  LYFT = "lyft",
  TASKRABBIT = "taskrabbit",
  MULTI_PLATFORM = "multi_platform",
}

/**
 * Metro area types - matches backend MetroArea enum
 */
export enum MetroArea {
  NEW_YORK = "new_york",
  LOS_ANGELES = "los_angeles",
  CHICAGO = "chicago",
  SAN_FRANCISCO = "san_francisco",
  WASHINGTON_DC = "washington_dc",
  RICHMOND = "richmond",
}

/**
 * Display names for platforms in UI dropdowns
 */
export const PLATFORM_DISPLAY_NAMES: Record<Platform, string> = {
  [Platform.UBER_RIDESHARE]: "Uber (Rideshare)",
  [Platform.UBER_EATS]: "Uber Eats",
  [Platform.DOORDASH]: "DoorDash",
  [Platform.INSTACART]: "Instacart",
  [Platform.LYFT]: "Lyft",
  [Platform.TASKRABBIT]: "TaskRabbit",
  [Platform.MULTI_PLATFORM]: "Multiple Platforms",
};

/**
 * Display names for metro areas in UI dropdowns
 */
export const METRO_DISPLAY_NAMES: Record<MetroArea, string> = {
  [MetroArea.NEW_YORK]: "New York, NY",
  [MetroArea.LOS_ANGELES]: "Los Angeles, CA",
  [MetroArea.CHICAGO]: "Chicago, IL",
  [MetroArea.SAN_FRANCISCO]: "San Francisco, CA",
  [MetroArea.WASHINGTON_DC]: "Washington, DC",
  [MetroArea.RICHMOND]: "Richmond, VA",
};

/**
 * Gig worker profile - matches backend GigWorkerProfile schema
 */
export interface GigWorkerProfile {
  platform: Platform;
  metro: MetroArea;
  hoursPerWeek: number; // 5-70 range
  monthsExperience: number; // affects earnings stability
  hasSecondaryIncome: boolean;
  secondaryMonthlyIncome: number; // W-2 or other stable income
  monthlyRent: number;
  monthlyFixedExpenses: number; // insurance, phone, subscriptions
  currentSavings: number;
  dependents: number;
}

/**
 * Default profile for form initialization
 */
export const DEFAULT_PROFILE: GigWorkerProfile = {
  platform: Platform.UBER_RIDESHARE,
  metro: MetroArea.WASHINGTON_DC,
  hoursPerWeek: 30,
  monthsExperience: 12,
  hasSecondaryIncome: false,
  secondaryMonthlyIncome: 0,
  monthlyRent: 1500,
  monthlyFixedExpenses: 500,
  currentSavings: 3000,
  dependents: 0,
};

/**
 * Validation constraints for profile fields
 */
export const PROFILE_CONSTRAINTS = {
  hoursPerWeek: { min: 5, max: 70 },
  monthsExperience: { min: 0, max: 240 },
  secondaryMonthlyIncome: { min: 0, max: 50000 },
  monthlyRent: { min: 0, max: 10000 },
  monthlyFixedExpenses: { min: 0, max: 10000 },
  currentSavings: { min: 0, max: 500000 },
  dependents: { min: 0, max: 10 },
};
