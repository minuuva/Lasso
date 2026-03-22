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
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useSimulation } from "@/contexts/SimulationContext";
import {
  Platform,
  MetroArea,
  PLATFORM_DISPLAY_NAMES,
  METRO_DISPLAY_NAMES,
  PROFILE_CONSTRAINTS,
} from "@/types";
import { ARCHETYPES } from "@/lib/constants";
import { archetypeToProfile } from "@/mocks/generators";
import { formatCurrency } from "@/lib/utils";
import { Car, MapPin, Clock, DollarSign, Users, UserCircle } from "lucide-react";

export function ProfileForm() {
  const { state, updateProfile, runSimulation } = useSimulation();
  const { profile } = state;

  // Load archetype as profile
  const handleArchetypeSelect = (archetypeId: string) => {
    const archetype = ARCHETYPES.find((a) => a.id === archetypeId);
    if (archetype) {
      const newProfile = archetypeToProfile(archetype);
      // Update all profile fields
      updateProfile(newProfile);
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Car className="h-5 w-5 text-primary" />
          Gig Worker Profile
        </CardTitle>
        <CardDescription>
          Enter your work details and financial situation
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Archetype Quick Select */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2">
            <UserCircle className="h-4 w-4" />
            Quick Load Persona
          </Label>
          <div className="grid grid-cols-1 gap-2">
            {ARCHETYPES.map((archetype) => (
              <Button
                key={archetype.id}
                variant="ghost"
                size="sm"
                className="justify-start h-auto py-2 px-3 text-left hover:bg-muted"
                onClick={() => handleArchetypeSelect(archetype.id)}
              >
                <div className="flex items-center justify-between w-full gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{archetype.name}</p>
                    <p className="text-xs text-muted-foreground truncate">
                      {archetype.description}
                    </p>
                  </div>
                  <Badge
                    variant={
                      archetype.defaultRiskCategory === "low"
                        ? "success"
                        : archetype.defaultRiskCategory === "medium"
                        ? "warning"
                        : "destructive"
                    }
                    className="shrink-0 text-[10px]"
                  >
                    {archetype.defaultRiskCategory}
                  </Badge>
                </div>
              </Button>
            ))}
          </div>
        </div>

        <div className="border-t border-border pt-4" />

        {/* Platform Selection */}
        <div className="space-y-2">
          <Label htmlFor="platform" className="flex items-center gap-2">
            <Car className="h-4 w-4" />
            Platform
          </Label>
          <Select
            value={profile.platform}
            onValueChange={(value: Platform) =>
              updateProfile({ platform: value })
            }
          >
            <SelectTrigger id="platform">
              <SelectValue placeholder="Select platform" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(PLATFORM_DISPLAY_NAMES).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Metro Area Selection */}
        <div className="space-y-2">
          <Label htmlFor="metro" className="flex items-center gap-2">
            <MapPin className="h-4 w-4" />
            Metro Area
          </Label>
          <Select
            value={profile.metro}
            onValueChange={(value: MetroArea) =>
              updateProfile({ metro: value })
            }
          >
            <SelectTrigger id="metro">
              <SelectValue placeholder="Select metro area" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(METRO_DISPLAY_NAMES).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Hours per Week Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Hours per Week
            </Label>
            <span className="text-sm font-medium text-primary">
              {profile.hoursPerWeek} hrs
            </span>
          </div>
          <Slider
            value={[profile.hoursPerWeek]}
            onValueChange={([value]) =>
              updateProfile({ hoursPerWeek: value })
            }
            min={PROFILE_CONSTRAINTS.hoursPerWeek.min}
            max={PROFILE_CONSTRAINTS.hoursPerWeek.max}
            step={1}
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>5 hrs</span>
            <span>70 hrs</span>
          </div>
        </div>

        {/* Experience */}
        <div className="space-y-2">
          <Label htmlFor="experience">Months of Experience</Label>
          <Input
            id="experience"
            type="number"
            value={profile.monthsExperience}
            onChange={(e) =>
              updateProfile({
                monthsExperience: Math.max(
                  0,
                  parseInt(e.target.value) || 0
                ),
              })
            }
            min={0}
            max={240}
          />
        </div>

        {/* Financial Section */}
        <div className="space-y-4 pt-2 border-t border-border">
          <h4 className="font-medium text-sm flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Monthly Finances
          </h4>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="rent" className="text-xs">
                Rent
              </Label>
              <Input
                id="rent"
                type="number"
                value={profile.monthlyRent}
                onChange={(e) =>
                  updateProfile({
                    monthlyRent: Math.max(0, parseInt(e.target.value) || 0),
                  })
                }
                min={0}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="expenses" className="text-xs">
                Fixed Expenses
              </Label>
              <Input
                id="expenses"
                type="number"
                value={profile.monthlyFixedExpenses}
                onChange={(e) =>
                  updateProfile({
                    monthlyFixedExpenses: Math.max(
                      0,
                      parseInt(e.target.value) || 0
                    ),
                  })
                }
                min={0}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="savings">Current Savings</Label>
            <Input
              id="savings"
              type="number"
              value={profile.currentSavings}
              onChange={(e) =>
                updateProfile({
                  currentSavings: Math.max(0, parseInt(e.target.value) || 0),
                })
              }
              min={0}
            />
          </div>
        </div>

        {/* Secondary Income Toggle */}
        <div className="space-y-3 pt-2 border-t border-border">
          <div className="flex items-center justify-between">
            <Label htmlFor="secondary" className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              Secondary Income (W-2, etc.)
            </Label>
            <Switch
              id="secondary"
              checked={profile.hasSecondaryIncome}
              onCheckedChange={(checked) =>
                updateProfile({ hasSecondaryIncome: checked })
              }
            />
          </div>
          {profile.hasSecondaryIncome && (
            <div className="space-y-2">
              <Label htmlFor="secondaryAmount" className="text-xs">
                Monthly Amount
              </Label>
              <Input
                id="secondaryAmount"
                type="number"
                value={profile.secondaryMonthlyIncome}
                onChange={(e) =>
                  updateProfile({
                    secondaryMonthlyIncome: Math.max(
                      0,
                      parseInt(e.target.value) || 0
                    ),
                  })
                }
                min={0}
              />
            </div>
          )}
        </div>

        {/* Dependents */}
        <div className="space-y-2">
          <Label htmlFor="dependents">Number of Dependents</Label>
          <Input
            id="dependents"
            type="number"
            value={profile.dependents}
            onChange={(e) =>
              updateProfile({
                dependents: Math.max(0, parseInt(e.target.value) || 0),
              })
            }
            min={0}
            max={10}
          />
        </div>

        {/* Run Simulation Button */}
        <Button
          onClick={runSimulation}
          disabled={state.isLoading}
          className="w-full mt-4"
          size="lg"
        >
          {state.isLoading ? (
            <>
              <span className="animate-pulse">Running Simulation...</span>
            </>
          ) : (
            "Run Simulation"
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

export default ProfileForm;
