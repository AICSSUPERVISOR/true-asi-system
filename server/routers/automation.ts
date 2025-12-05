/**
 * Recommendation Automation tRPC Router
 * 
 * Exposes recommendation automation functionality via tRPC API
 */

import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import {
  Recommendation,
  ExecutionPlan,
  generateExecutionPlan,
  batchGenerateExecutionPlans,
  findMatchingPlatforms,
  getAutomationStats,
  parseRecommendation,
} from "../helpers/recommendation_automation";

/**
 * Recommendation schema
 */
const recommendationSchema = z.object({
  id: z.string(),
  category: z.enum([
    'pricing',
    'marketing',
    'operations',
    'technology',
    'hr',
    'sales',
    'customer_service',
    'finance',
    'legal',
    'product',
  ]),
  impact: z.enum(['low', 'medium', 'high']),
  difficulty: z.enum(['easy', 'medium', 'hard']),
  title: z.string(),
  description: z.string(),
  expectedROI: z.string(),
  priority: z.number().min(1).max(10),
  cost: z.string(),
  timeframe: z.string(),
  isAutomated: z.boolean(),
});

/**
 * Automation router
 */
export const automationRouter = router({
  /**
   * Generate execution plan for a single recommendation
   */
  generateExecutionPlan: publicProcedure
    .input(recommendationSchema)
    .query(async ({ input }) => {
      try {
        const plan = generateExecutionPlan(input as Recommendation);
        
        return {
          success: true,
          plan,
        };
      } catch (error) {
        console.error('[Automation] Error generating execution plan:', error);
        return {
          success: false,
          error: 'Failed to generate execution plan',
        };
      }
    }),

  /**
   * Batch generate execution plans for multiple recommendations
   */
  batchGenerateExecutionPlans: publicProcedure
    .input(z.object({
      recommendations: z.array(recommendationSchema),
    }))
    .query(async ({ input }) => {
      try {
        const plans = batchGenerateExecutionPlans(input.recommendations as Recommendation[]);
        const stats = getAutomationStats(plans);
        
        return {
          success: true,
          plans,
          stats,
        };
      } catch (error) {
        console.error('[Automation] Error batch generating execution plans:', error);
        return {
          success: false,
          error: 'Failed to batch generate execution plans',
        };
      }
    }),

  /**
   * Find matching platforms for a recommendation
   */
  findMatchingPlatforms: publicProcedure
    .input(recommendationSchema)
    .query(async ({ input }) => {
      try {
        const platforms = findMatchingPlatforms(input as Recommendation);
        
        return {
          success: true,
          platforms,
          count: platforms.length,
        };
      } catch (error) {
        console.error('[Automation] Error finding matching platforms:', error);
        return {
          success: false,
          error: 'Failed to find matching platforms',
        };
      }
    }),

  /**
   * Parse recommendation text (Capgemini format)
   */
  parseRecommendation: publicProcedure
    .input(z.object({
      text: z.string(),
    }))
    .query(async ({ input }) => {
      try {
        const recommendation = parseRecommendation(input.text);
        
        return {
          success: true,
          recommendation,
        };
      } catch (error) {
        console.error('[Automation] Error parsing recommendation:', error);
        return {
          success: false,
          error: 'Failed to parse recommendation',
        };
      }
    }),

  /**
   * Get automation statistics for a set of recommendations
   */
  getAutomationStats: publicProcedure
    .input(z.object({
      recommendations: z.array(recommendationSchema),
    }))
    .query(async ({ input }) => {
      try {
        const plans = batchGenerateExecutionPlans(input.recommendations as Recommendation[]);
        const stats = getAutomationStats(plans);
        
        return {
          success: true,
          stats,
        };
      } catch (error) {
        console.error('[Automation] Error getting automation stats:', error);
        return {
          success: false,
          error: 'Failed to get automation stats',
        };
      }
    }),

  /**
   * Convert Capgemini recommendations to executable plans
   */
  convertCapgeminiRecommendations: publicProcedure
    .input(z.object({
      recommendationsText: z.string(),
    }))
    .query(async ({ input }) => {
      try {
        // Split text into individual recommendations
        const recommendationTexts = input.recommendationsText
          .split(/\n\n+/)
          .filter(text => text.trim().length > 0);

        // Parse each recommendation
        const recommendations = recommendationTexts.map(text => parseRecommendation(text));

        // Generate execution plans
        const plans = batchGenerateExecutionPlans(recommendations);
        const stats = getAutomationStats(plans);

        return {
          success: true,
          recommendations,
          plans,
          stats,
          message: `Converted ${recommendations.length} recommendations. ${stats.partiallyAutomated + stats.fullyAutomated} can be automated (${stats.coveragePercentage.toFixed(1)}% coverage).`,
        };
      } catch (error) {
        console.error('[Automation] Error converting Capgemini recommendations:', error);
        return {
          success: false,
          error: 'Failed to convert recommendations',
        };
      }
    }),
});
