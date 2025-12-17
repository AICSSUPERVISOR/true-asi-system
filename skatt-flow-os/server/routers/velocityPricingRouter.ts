import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { publicProcedure, protectedProcedure, router } from "../_core/trpc";
import { nanoid } from "nanoid";

// ============================================================================
// VELOCITY PRICING ROUTER
// Handles competitor price monitoring and VELOCITY pricing calculations
// ============================================================================

// Zod schemas for input validation
const competitorPriceSchema = z.object({
  competitor: z.enum(["Uber", "Bolt", "Lyft"]),
  route: z.string(),
  cityFrom: z.string(),
  cityTo: z.string(),
  currency: z.enum(["GBP", "USD", "EUR"]),
  standardPrice: z.number().positive(),
  premiumPrice: z.number().positive().optional(),
  xlPrice: z.number().positive().optional(),
  sourceUrl: z.string().url().optional(),
});

const velocityPriceSchema = z.object({
  route: z.string(),
  cityFrom: z.string(),
  cityTo: z.string(),
  currency: z.enum(["GBP", "USD", "EUR"]),
  velocityStandard: z.number().positive(),
  velocityPremium: z.number().positive().optional(),
  velocityXl: z.number().positive().optional(),
  velocityAutonomous: z.number().positive(),
  uberPrice: z.number().positive().optional(),
  boltPrice: z.number().positive().optional(),
  lyftPrice: z.number().positive().optional(),
  uberDiscount: z.number().default(10.0),
  boltDiscount: z.number().default(8.0),
  lyftDiscount: z.number().default(10.0),
  autonomousDiscount: z.number().default(40.0),
  priceChangePercent: z.number().default(0),
  significantChange: z.boolean().default(false),
});

const priceMonitoringResultSchema = z.object({
  timestamp: z.string(),
  competitorPrices: z.array(competitorPriceSchema),
  velocityPrices: z.array(velocityPriceSchema),
  slackMessageTimestamp: z.string().optional(),
});

// In-memory storage for demo (replace with actual database operations)
let competitorPricesStore: z.infer<typeof competitorPriceSchema>[] = [];
let velocityPricesStore: z.infer<typeof velocityPriceSchema>[] = [];
let monitoringLogsStore: {
  runId: string;
  status: string;
  routesProcessed: number;
  competitorPricesFound: number;
  velocityPricesCalculated: number;
  slackNotificationSent: boolean;
  slackMessageTimestamp?: string;
  startedAt: Date;
  completedAt?: Date;
  rawResultsJson?: unknown;
}[] = [];

export const velocityPricingRouter = router({
  // ============================================================================
  // COMPETITOR PRICES
  // ============================================================================

  /**
   * Store competitor prices from scraping results
   */
  storeCompetitorPrices: protectedProcedure
    .input(z.object({
      prices: z.array(competitorPriceSchema),
    }))
    .mutation(async ({ input }) => {
      const timestamp = new Date().toISOString();
      
      // Add scraped timestamp to each price
      const pricesWithTimestamp = input.prices.map(price => ({
        ...price,
        scrapedAt: timestamp,
      }));

      // Store prices (in production, this would be a database insert)
      competitorPricesStore = [...competitorPricesStore, ...pricesWithTimestamp];

      return {
        success: true,
        count: input.prices.length,
        timestamp,
      };
    }),

  /**
   * Get latest competitor prices for a route
   */
  getCompetitorPrices: publicProcedure
    .input(z.object({
      route: z.string().optional(),
      competitor: z.enum(["Uber", "Bolt", "Lyft"]).optional(),
    }))
    .query(async ({ input }) => {
      let prices = competitorPricesStore;

      if (input.route) {
        prices = prices.filter(p => p.route === input.route);
      }
      if (input.competitor) {
        prices = prices.filter(p => p.competitor === input.competitor);
      }

      return prices;
    }),

  // ============================================================================
  // VELOCITY PRICES
  // ============================================================================

  /**
   * Store calculated VELOCITY prices
   */
  storeVelocityPrices: protectedProcedure
    .input(z.object({
      prices: z.array(velocityPriceSchema),
    }))
    .mutation(async ({ input }) => {
      const timestamp = new Date().toISOString();

      // Store prices (in production, this would be a database upsert)
      velocityPricesStore = input.prices.map(price => ({
        ...price,
        calculatedAt: timestamp,
      }));

      return {
        success: true,
        count: input.prices.length,
        timestamp,
      };
    }),

  /**
   * Get current VELOCITY prices
   */
  getVelocityPrices: publicProcedure
    .input(z.object({
      route: z.string().optional(),
    }))
    .query(async ({ input }) => {
      let prices = velocityPricesStore;

      if (input.route) {
        prices = prices.filter(p => p.route === input.route);
      }

      return prices;
    }),

  // ============================================================================
  // PRICE MONITORING
  // ============================================================================

  /**
   * Store complete price monitoring run results
   */
  storeMonitoringResults: protectedProcedure
    .input(priceMonitoringResultSchema.extend({
      slackMessageTimestamp: z.string().optional(),
    }))
    .mutation(async ({ input }) => {
      const runId = nanoid();
      const startedAt = new Date();

      // Store competitor prices
      competitorPricesStore = input.competitorPrices;

      // Store velocity prices
      velocityPricesStore = input.velocityPrices;

      // Create monitoring log
      const log = {
        runId,
        status: "COMPLETED",
        routesProcessed: input.velocityPrices.length,
        competitorPricesFound: input.competitorPrices.length,
        velocityPricesCalculated: input.velocityPrices.length,
        slackNotificationSent: !!input.slackMessageTimestamp,
        slackMessageTimestamp: input.slackMessageTimestamp,
        startedAt,
        completedAt: new Date(),
        rawResultsJson: input,
      };

      monitoringLogsStore.push(log);

      return {
        success: true,
        runId,
        summary: {
          competitorPricesStored: input.competitorPrices.length,
          velocityPricesCalculated: input.velocityPrices.length,
          slackNotificationSent: !!input.slackMessageTimestamp,
        },
      };
    }),

  /**
   * Get price monitoring history
   */
  getMonitoringHistory: protectedProcedure
    .input(z.object({
      limit: z.number().min(1).max(100).default(10),
    }))
    .query(async ({ input }) => {
      return monitoringLogsStore
        .slice(-input.limit)
        .reverse();
    }),

  /**
   * Get pricing analytics summary
   */
  getPricingAnalytics: publicProcedure.query(async () => {
    const routes = [...new Set(velocityPricesStore.map(p => p.route))];
    
    const analytics = routes.map(route => {
      const routePrices = velocityPricesStore.filter(p => p.route === route);
      const latestPrice = routePrices[routePrices.length - 1];
      
      if (!latestPrice) return null;

      const avgCompetitorPrice = [
        latestPrice.uberPrice,
        latestPrice.boltPrice,
        latestPrice.lyftPrice,
      ].filter(Boolean).reduce((sum, p) => sum + (p || 0), 0) / 
        [latestPrice.uberPrice, latestPrice.boltPrice, latestPrice.lyftPrice].filter(Boolean).length;

      return {
        route,
        currency: latestPrice.currency,
        velocityStandard: latestPrice.velocityStandard,
        velocityAutonomous: latestPrice.velocityAutonomous,
        avgCompetitorPrice,
        savingsPercent: ((avgCompetitorPrice - latestPrice.velocityStandard) / avgCompetitorPrice * 100).toFixed(1),
        autonomousSavingsPercent: ((avgCompetitorPrice - latestPrice.velocityAutonomous) / avgCompetitorPrice * 100).toFixed(1),
      };
    }).filter(Boolean);

    return {
      totalRoutes: routes.length,
      lastUpdated: monitoringLogsStore[monitoringLogsStore.length - 1]?.completedAt?.toISOString(),
      analytics,
    };
  }),

  // ============================================================================
  // PRICE COMPARISON
  // ============================================================================

  /**
   * Compare prices for a specific route
   */
  comparePrices: publicProcedure
    .input(z.object({
      route: z.string(),
    }))
    .query(async ({ input }) => {
      const competitorPrices = competitorPricesStore.filter(p => p.route === input.route);
      const velocityPrice = velocityPricesStore.find(p => p.route === input.route);

      if (!velocityPrice) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: `No pricing data found for route: ${input.route}`,
        });
      }

      const comparison = {
        route: input.route,
        currency: velocityPrice.currency,
        competitors: competitorPrices.map(cp => ({
          name: cp.competitor,
          standardPrice: cp.standardPrice,
          premiumPrice: cp.premiumPrice,
          xlPrice: cp.xlPrice,
        })),
        velocity: {
          standardPrice: velocityPrice.velocityStandard,
          premiumPrice: velocityPrice.velocityPremium,
          xlPrice: velocityPrice.velocityXl,
          autonomousPrice: velocityPrice.velocityAutonomous,
        },
        savings: {
          vsUber: velocityPrice.uberPrice 
            ? ((velocityPrice.uberPrice - velocityPrice.velocityStandard) / velocityPrice.uberPrice * 100).toFixed(1) + "%"
            : null,
          vsBolt: velocityPrice.boltPrice
            ? ((velocityPrice.boltPrice - velocityPrice.velocityStandard) / velocityPrice.boltPrice * 100).toFixed(1) + "%"
            : null,
          vsLyft: velocityPrice.lyftPrice
            ? ((velocityPrice.lyftPrice - velocityPrice.velocityStandard) / velocityPrice.lyftPrice * 100).toFixed(1) + "%"
            : null,
          autonomousVsMarket: velocityPrice.velocityAutonomous
            ? "40%"
            : null,
        },
      };

      return comparison;
    }),
});

export type VelocityPricingRouter = typeof velocityPricingRouter;
