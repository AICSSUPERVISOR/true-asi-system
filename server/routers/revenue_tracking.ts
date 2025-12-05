import { z } from "zod";
import { publicProcedure, protectedProcedure, router } from "../_core/trpc";
import { getDb } from "../db";
import { revenueTracking, analyses } from "../../drizzle/schema";
import { eq, desc, and, gte, lte } from "drizzle-orm";
import { TRPCError } from "@trpc/server";

export const revenueTrackingRouter = router({
  /**
   * Track metrics for a period
   */
  trackMetrics: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
      executionId: z.string().optional(),
      revenue: z.number().optional(),
      customers: z.number().optional(),
      newCustomers: z.number().optional(),
      websiteTraffic: z.number().optional(),
      websiteConversionRate: z.number().optional(), // Percentage (e.g., 2.5 for 2.5%)
      linkedinFollowers: z.number().optional(),
      linkedinEngagement: z.number().optional(), // Percentage
      socialMediaFollowers: z.number().optional(),
      socialMediaEngagement: z.number().optional(), // Percentage
      averageRating: z.number().optional(), // Rating (e.g., 4.5)
      totalReviews: z.number().optional(),
      periodStart: z.string(), // ISO timestamp
      periodEnd: z.string(), // ISO timestamp
    }))
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      // Verify analysis ownership
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis || analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to track metrics for this analysis",
        });
      }

      const trackingId = `tracking_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      await db.insert(revenueTracking).values({
        id: trackingId,
        analysisId: input.analysisId,
        executionId: input.executionId || null,
        revenue: input.revenue || null,
        customers: input.customers || null,
        newCustomers: input.newCustomers || null,
        websiteTraffic: input.websiteTraffic || null,
        websiteConversionRate: input.websiteConversionRate ? Math.round(input.websiteConversionRate * 10) : null,
        linkedinFollowers: input.linkedinFollowers || null,
        linkedinEngagement: input.linkedinEngagement ? Math.round(input.linkedinEngagement * 10) : null,
        socialMediaFollowers: input.socialMediaFollowers || null,
        socialMediaEngagement: input.socialMediaEngagement ? Math.round(input.socialMediaEngagement * 10) : null,
        averageRating: input.averageRating ? Math.round(input.averageRating * 10) : null,
        totalReviews: input.totalReviews || null,
        periodStart: new Date(input.periodStart),
        periodEnd: new Date(input.periodEnd),
      });

      return {
        trackingId,
        success: true,
      };
    }),

  /**
   * Get all tracking records for an analysis
   */
  getTrackingRecords: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
      startDate: z.string().optional(), // ISO date string
      endDate: z.string().optional(), // ISO date string
    }))
    .query(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      // Verify analysis ownership
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis || analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to view metrics for this analysis",
        });
      }

      // Build where conditions
      const conditions = [eq(revenueTracking.analysisId, input.analysisId)];
      
      if (input.startDate) {
        conditions.push(gte(revenueTracking.periodStart, new Date(input.startDate)));
      }
      
      if (input.endDate) {
        conditions.push(lte(revenueTracking.periodEnd, new Date(input.endDate)));
      }

      const records = await db
        .select()
        .from(revenueTracking)
        .where(and(...conditions))
        .orderBy(desc(revenueTracking.periodStart));

      return records.map(r => ({
        ...r,
        websiteConversionRate: r.websiteConversionRate ? r.websiteConversionRate / 10 : null,
        linkedinEngagement: r.linkedinEngagement ? r.linkedinEngagement / 10 : null,
        socialMediaEngagement: r.socialMediaEngagement ? r.socialMediaEngagement / 10 : null,
        averageRating: r.averageRating ? r.averageRating / 10 : null,
      }));
    }),

  /**
   * Calculate ROI for an analysis
   */
  calculateROI: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
    }))
    .query(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      // Verify analysis ownership
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis || analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to calculate ROI for this analysis",
        });
      }

      // Get all tracking records ordered by date
      const records = await db
        .select()
        .from(revenueTracking)
        .where(eq(revenueTracking.analysisId, input.analysisId))
        .orderBy(revenueTracking.periodStart);

      if (records.length < 2) {
        return {
          roi: 0,
          totalRevenueIncrease: 0,
          totalCustomerIncrease: 0,
          averageMonthlyGrowth: 0,
          message: "Not enough data to calculate ROI. Need at least 2 tracking records.",
        };
      }

      const firstRecord = records[0];
      const lastRecord = records[records.length - 1];

      // Calculate revenue increase
      const firstRevenue = firstRecord.revenue || 0;
      const lastRevenue = lastRecord.revenue || 0;
      const totalRevenueIncrease = lastRevenue - firstRevenue;

      // Calculate customer increase
      const firstCustomers = firstRecord.customers || 0;
      const lastCustomers = lastRecord.customers || 0;
      const totalCustomerIncrease = lastCustomers - firstCustomers;

      // Calculate time period in months
      const firstDate = new Date(firstRecord.periodStart);
      const lastDate = new Date(lastRecord.periodEnd);
      const monthsDiff = (lastDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24 * 30);
      const averageMonthlyGrowth = monthsDiff > 0 ? (totalRevenueIncrease / monthsDiff) : 0;

      // Calculate ROI (assuming initial investment is stored in analysis data)
      const analysisData = JSON.parse(analysis.analysisData);
      const totalInvestment = analysisData.totalCost || 10000; // Default to 10k if not specified
      const roi = totalInvestment > 0 ? (totalRevenueIncrease / totalInvestment) * 100 : 0;

      // Calculate other improvements
      const websiteTrafficIncrease = (lastRecord.websiteTraffic || 0) - (firstRecord.websiteTraffic || 0);
      const linkedinFollowersIncrease = (lastRecord.linkedinFollowers || 0) - (firstRecord.linkedinFollowers || 0);
      const socialMediaFollowersIncrease = (lastRecord.socialMediaFollowers || 0) - (firstRecord.socialMediaFollowers || 0);
      const reviewsIncrease = (lastRecord.totalReviews || 0) - (firstRecord.totalReviews || 0);

      return {
        roi: parseFloat(roi.toFixed(2)),
        totalRevenueIncrease,
        totalCustomerIncrease,
        averageMonthlyGrowth: parseFloat(averageMonthlyGrowth.toFixed(2)),
        totalInvestment,
        firstRevenue,
        lastRevenue,
        firstCustomers,
        lastCustomers,
        websiteTrafficIncrease,
        linkedinFollowersIncrease,
        socialMediaFollowersIncrease,
        reviewsIncrease,
        measurementPeriodMonths: parseFloat(monthsDiff.toFixed(1)),
        dataPoints: records.length,
      };
    }),

  /**
   * Get latest metrics summary
   */
  getLatestMetrics: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
    }))
    .query(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      // Verify analysis ownership
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis || analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to view metrics for this analysis",
        });
      }

      // Get latest tracking record
      const [latestRecord] = await db
        .select()
        .from(revenueTracking)
        .where(eq(revenueTracking.analysisId, input.analysisId))
        .orderBy(desc(revenueTracking.periodStart))
        .limit(1);

      if (!latestRecord) {
        return null;
      }

      // Get previous record for comparison
      const [previousRecord] = await db
        .select()
        .from(revenueTracking)
        .where(eq(revenueTracking.analysisId, input.analysisId))
        .orderBy(desc(revenueTracking.periodStart))
        .limit(1)
        .offset(1);

      // Calculate changes
      const calculateChange = (current: number | null, previous: number | null) => {
        if (!current || !previous || previous === 0) return null;
        return parseFloat((((current - previous) / previous) * 100).toFixed(2));
      };

      return {
        current: {
          revenue: latestRecord.revenue,
          customers: latestRecord.customers,
          newCustomers: latestRecord.newCustomers,
          websiteTraffic: latestRecord.websiteTraffic,
          websiteConversionRate: latestRecord.websiteConversionRate ? latestRecord.websiteConversionRate / 10 : null,
          linkedinFollowers: latestRecord.linkedinFollowers,
          linkedinEngagement: latestRecord.linkedinEngagement ? latestRecord.linkedinEngagement / 10 : null,
          socialMediaFollowers: latestRecord.socialMediaFollowers,
          socialMediaEngagement: latestRecord.socialMediaEngagement ? latestRecord.socialMediaEngagement / 10 : null,
          averageRating: latestRecord.averageRating ? latestRecord.averageRating / 10 : null,
          totalReviews: latestRecord.totalReviews,
          periodStart: latestRecord.periodStart,
          periodEnd: latestRecord.periodEnd,
        },
        changes: previousRecord ? {
          revenue: calculateChange(latestRecord.revenue, previousRecord.revenue),
          customers: calculateChange(latestRecord.customers, previousRecord.customers),
          websiteTraffic: calculateChange(latestRecord.websiteTraffic, previousRecord.websiteTraffic),
          linkedinFollowers: calculateChange(latestRecord.linkedinFollowers, previousRecord.linkedinFollowers),
          socialMediaFollowers: calculateChange(latestRecord.socialMediaFollowers, previousRecord.socialMediaFollowers),
          totalReviews: calculateChange(latestRecord.totalReviews, previousRecord.totalReviews),
        } : null,
      };
    }),

  /**
   * Generate email report (weekly/monthly)
   */
  generateReport: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
      reportType: z.enum(["weekly", "monthly"]),
    }))
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) throw new TRPCError({ code: "INTERNAL_SERVER_ERROR", message: "Database not available" });

      // Verify analysis ownership
      const [analysis] = await db
        .select()
        .from(analyses)
        .where(eq(analyses.id, input.analysisId));

      if (!analysis || analysis.userId !== ctx.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to generate reports for this analysis",
        });
      }

      // Calculate date range
      const endDate = new Date();
      const startDate = new Date();
      if (input.reportType === "weekly") {
        startDate.setDate(startDate.getDate() - 7);
      } else {
        startDate.setMonth(startDate.getMonth() - 1);
      }

      // Get all records in date range
      const records = await db
        .select()
        .from(revenueTracking)
        .where(and(
          eq(revenueTracking.analysisId, input.analysisId),
          gte(revenueTracking.periodStart, startDate),
          lte(revenueTracking.periodEnd, endDate)
        ))
        .orderBy(revenueTracking.periodStart);

      if (records.length < 2) {
        return {
          report: null,
          emailSent: false,
          message: "Not enough data to generate report. Need at least 2 tracking records in the period.",
        };
      }

      const firstRecord = records[0];
      const lastRecord = records[records.length - 1];

      // Calculate summary
      const summary = {
        revenue: {
          start: firstRecord.revenue || 0,
          end: lastRecord.revenue || 0,
          change: (lastRecord.revenue || 0) - (firstRecord.revenue || 0),
          changePercentage: firstRecord.revenue ? (((lastRecord.revenue || 0) - (firstRecord.revenue || 0)) / (firstRecord.revenue || 1)) * 100 : 0,
        },
        customers: {
          start: firstRecord.customers || 0,
          end: lastRecord.customers || 0,
          change: (lastRecord.customers || 0) - (firstRecord.customers || 0),
          changePercentage: firstRecord.customers ? (((lastRecord.customers || 0) - (firstRecord.customers || 0)) / (firstRecord.customers || 1)) * 100 : 0,
        },
        websiteTraffic: {
          start: firstRecord.websiteTraffic || 0,
          end: lastRecord.websiteTraffic || 0,
          change: (lastRecord.websiteTraffic || 0) - (firstRecord.websiteTraffic || 0),
          changePercentage: firstRecord.websiteTraffic ? (((lastRecord.websiteTraffic || 0) - (firstRecord.websiteTraffic || 0)) / (firstRecord.websiteTraffic || 1)) * 100 : 0,
        },
        linkedinFollowers: {
          start: firstRecord.linkedinFollowers || 0,
          end: lastRecord.linkedinFollowers || 0,
          change: (lastRecord.linkedinFollowers || 0) - (firstRecord.linkedinFollowers || 0),
          changePercentage: firstRecord.linkedinFollowers ? (((lastRecord.linkedinFollowers || 0) - (firstRecord.linkedinFollowers || 0)) / (firstRecord.linkedinFollowers || 1)) * 100 : 0,
        },
      };

      // Generate report object (would be sent via email in production)
      const report = {
        companyName: analysis.companyName,
        reportType: input.reportType,
        period: {
          start: startDate.toISOString(),
          end: endDate.toISOString(),
        },
        summary,
        totalDataPoints: records.length,
        generatedAt: new Date().toISOString(),
      };

      // TODO: Send email via SendGrid/AWS SES
      // For now, just return the report data
      return {
        report,
        emailSent: false, // Would be true after email integration
        message: "Report generated successfully. Email integration pending.",
      };
    }),
});
